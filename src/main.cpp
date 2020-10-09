#include "main.h"

#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "args/args.hxx"
#include "imgui.h"
#include "surface_derivatives.h"

#include "energy/tpe_kernel.h"
#include "energy/all_pairs_tpe.h"
#include "energy/all_pairs_with_bvh.h"
#include "energy/barnes_hut_tpe_xdiff.h"
#include "energy/barnes_hut_newtonian.h"
#include "energy/static_obstacle.h"
#include "helpers.h"
#include <memory>
#include "spatial/bvh_flattened.h"

#include <Eigen/Sparse>
#include <omp.h>

#include "sobolev/all_constraints.h"
#include "sobolev/hs.h"
#include "sobolev/h1.h"
#include "spatial/convolution.h"
#include "spatial/convolution_kernel.h"
#include "block_cluster_tree.h"
#include "scene_file.h"
#include "surface_derivatives.h"

#include "remeshing/remeshing.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace rsurfaces
{

  MainApp *MainApp::instance = 0;

  MainApp::MainApp(MeshPtr mesh_, GeomPtr geom_, SurfaceFlow *flow_, polyscope::SurfaceMesh *psMesh_, std::string meshName_)
  {
    mesh = std::move(mesh_);
    geom = std::move(geom_);
    flow = flow_;
    psMesh = psMesh_;
    meshName = meshName_;
    vertBVH = 0;
  }

  void MainApp::TakeNaiveStep(double t)
  {
    flow->StepNaive(t);
  }

  void MainApp::TakeFractionalSobolevStep()
  {
    flow->StepFractionalSobolev();
  }

  void MainApp::updateMeshPositions()
  {
    if (normalizeView)
    {
      double scale = 0;
      for (GCVertex v : mesh->vertices())
      {
        scale = fmax(scale, norm(geom->inputVertexPositions[v]));
      }
      std::vector<Vector3> scaled(mesh->nVertices());
      VertexIndices inds = mesh->getVertexIndices();
      for (GCVertex v : mesh->vertices())
      {
        scaled[inds[v]] = geom->inputVertexPositions[v] / scale;
      }
      psMesh->updateVertexPositions(scaled);
    }
    else
    {
      psMesh->updateVertexPositions(geom->inputVertexPositions);
    }
    polyscope::requestRedraw();
  }

  void PlotMatrix(Eigen::MatrixXd &mat, polyscope::SurfaceMesh *psMesh, std::string name)
  {
    std::vector<Vector3> vecs;
    for (int i = 0; i < mat.rows(); i++)
    {
      Vector3 row_i = GetRow(mat, i);
      vecs.push_back(row_i);
    }
    psMesh->addVertexVectorQuantity(name, vecs);
  }

  void PlotVector(Eigen::VectorXd &vec, int nVerts, polyscope::SurfaceMesh *psMesh, std::string name)
  {
    Eigen::MatrixXd M;
    M.setZero(nVerts, 3);
    MatrixUtils::ColumnIntoMatrix(vec, M);
    PlotMatrix(M, psMesh, name);
  }

  void MainApp::TestLML()
  {
    SurfaceEnergy *energy = flow->BaseEnergy();
    Vector2 exps = energy->GetExponents();
    energy->Update();

    Eigen::MatrixXd gradient(mesh->nVertices(), 3);
    energy->Differential(gradient);
    Eigen::MatrixXd result = gradient;

    Hs::HsMetric hs(energy);

    hs.ProjectViaSparseMat(gradient, result);

    std::cout << result << std::endl;
    PlotMatrix(result, psMesh, "LML approx");
  }

  void MainApp::TestBarnesHut()
  {
    TPEKernel *tpe = new rsurfaces::TPEKernel(mesh, geom, 6, 12);
    SurfaceEnergy *energy_ap, *energy_bh, *energy_newton;
    energy_ap = new AllPairsTPEnergy(tpe);
    energy_bh = new BarnesHutTPEnergy6D(tpe, bh_theta);
    energy_newton = new BarnesHutNewtonian(tpe, bh_theta);

    energy_ap->Update();
    energy_bh->Update();
    energy_newton->Update();

    bool doAllPairs = (mesh->nFaces() < 3000);

    Eigen::MatrixXd grad_ap(mesh->nVertices(), 3);
    Eigen::MatrixXd grad_bh(mesh->nVertices(), 3);
    Eigen::MatrixXd grad_newton(mesh->nVertices(), 3);
    grad_ap.setZero();
    grad_bh.setZero();
    grad_newton.setZero();

    long start_ape = currentTimeMilliseconds();
    double value_ap = 0;
    if (doAllPairs)
      energy_ap->Value();
    long end_ape = currentTimeMilliseconds();

    long start_bhe = currentTimeMilliseconds();
    double value_bh = energy_bh->Value();
    long end_bhe = currentTimeMilliseconds();

    double val_error = fabs(value_ap - value_bh) / value_ap;

    if (!doAllPairs)
      std::cout << "Mesh has too many faces; not running all-pairs comparison." << std::endl;

    std::cout << "\n=====   Energy   =====" << std::endl;
    if (doAllPairs)
      std::cout << "All-pairs energy value  = " << value_ap << std::endl;
    std::cout << "Barnes-Hut energy value = " << value_bh << std::endl;
    if (doAllPairs)
      std::cout << "Relative error     = " << val_error * 100 << " percent" << std::endl;
    if (doAllPairs)
      std::cout << "All-pairs time     = " << (end_ape - start_ape) << " ms" << std::endl;
    std::cout << "Barnes-Hut time    = " << (end_bhe - start_bhe) << " ms" << std::endl;

    long start_apg = currentTimeMilliseconds();
    if (doAllPairs)
      energy_ap->Differential(grad_ap);
    long end_apg = currentTimeMilliseconds();

    long start_bhg = currentTimeMilliseconds();
    energy_bh->Differential(grad_bh);
    long end_bhg = currentTimeMilliseconds();

    long start_newton = currentTimeMilliseconds();
    energy_newton->Differential(grad_newton);
    long end_newton = currentTimeMilliseconds();

    double grad_error = (grad_ap - grad_bh).norm() / grad_ap.norm();

    std::cout << "\n=====  Gradient  =====" << std::endl;
    if (doAllPairs)
      std::cout << "All-pairs gradient norm      = " << grad_ap.norm() << std::endl;
    std::cout << "Barnes-Hut gradient norm     = " << grad_bh.norm() << std::endl;
    std::cout << "Newton gravity norm          = " << grad_newton.norm() << std::endl;
    if (doAllPairs)
      std::cout << "Barnes-Hut relative error    = " << grad_error * 100 << " percent" << std::endl;
    if (doAllPairs)
      std::cout << "All-pairs time     = " << (end_apg - start_apg) << " ms" << std::endl;
    std::cout << "Barnes-Hut time    = " << (end_bhg - start_bhg) << " ms" << std::endl;
    std::cout << "Newtonian time     = " << (end_newton - start_newton) << " ms" << std::endl;

    delete energy_ap;
    delete energy_bh;
    delete energy_newton;
    delete tpe;
  }

  void MainApp::PlotEnergyPerFace()
  {
    TPEKernel *tpe = new rsurfaces::TPEKernel(mesh, geom, 6, 12);
    BarnesHutTPEnergy6D *energy_bh = new BarnesHutTPEnergy6D(tpe, bh_theta);

    energy_bh->Update();
    double total = energy_bh->Value();

    for (GCFace f : mesh->faces())
    {
      double e = energy_bh->energyPerFace[f];
      // This looks like it scales the right way:
      // doubling the mesh also doubles the resulting lengths
      energy_bh->energyPerFace[f] = pow(e, 1.0 / (2 - tpe->alpha));
    }

    psMesh->addFaceScalarQuantity("energy per face", energy_bh->energyPerFace);
    std::cout << "Total energy = " << total << std::endl;

    delete energy_bh;
    delete tpe;
  }

  void MainApp::Scale2x()
  {
    for (GCVertex v : mesh->vertices())
    {
      geom->inputVertexPositions[v] = 2 * geom->inputVertexPositions[v];
    }
  }

  Jacobian numericalNormalDeriv(GeomPtr &geom, GCVertex vert, GCVertex wrt)
  {
    double h = 1e-4;

    Vector3 origNormal = vertexAreaNormalUnnormalized(geom, vert);
    Vector3 origPos = geom->inputVertexPositions[wrt];
    geom->inputVertexPositions[wrt] = origPos + Vector3{h, 0, 0};
    geom->refreshQuantities();
    Vector3 n_x = vertexAreaNormalUnnormalized(geom, vert);

    geom->inputVertexPositions[wrt] = origPos + Vector3{0, h, 0};
    geom->refreshQuantities();
    Vector3 n_y = vertexAreaNormalUnnormalized(geom, vert);

    geom->inputVertexPositions[wrt] = origPos + Vector3{0, 0, h};
    geom->refreshQuantities();
    Vector3 n_z = vertexAreaNormalUnnormalized(geom, vert);

    geom->inputVertexPositions[wrt] = origPos;
    geom->refreshQuantities();

    Vector3 deriv_y = (n_y - origNormal) / h;
    Vector3 deriv_z = (n_z - origNormal) / h;
    Vector3 deriv_x = (n_x - origNormal) / h;
    Jacobian J_num{deriv_x, deriv_y, deriv_z};
    return J_num;
  }

  void MainApp::TestNormalDeriv()
  {
    GCVertex vert;
    for (GCVertex v : mesh->vertices())
    {
      if (v.isBoundary())
      {
        vert = v;
        break;
      }
    }
    std::cout << "Testing vertex " << vert << std::endl;
    for (GCVertex neighbor : vert.adjacentVertices())
    {
      std::cout << "Derivative of normal of " << vert << " wrt " << neighbor << std::endl;
      Jacobian dWrtNeighbor = SurfaceDerivs::vertexNormalWrtVertex(geom, vert, neighbor);
      dWrtNeighbor.Print();
      std::cout << "Numerical:" << std::endl;
      numericalNormalDeriv(geom, vert, neighbor).Print();
    }
    std::cout << "Derivative of normal of " << vert << " wrt " << vert << std::endl;
    Jacobian dWrtSelf = SurfaceDerivs::vertexNormalWrtVertex(geom, vert, vert);
    dWrtSelf.Print();
    std::cout << "Numerical:" << std::endl;
    numericalNormalDeriv(geom, vert, vert).Print();
  }

  void MainApp::TestMVProduct()
  {
    long gradientStartTime = currentTimeMilliseconds();
    // Use the differential of the energy as the test case
    SurfaceEnergy *energy = flow->BaseEnergy();
    energy->Update();
    Eigen::MatrixXd gradient(mesh->nVertices(), 3);
    energy->Differential(gradient);
    Eigen::VectorXd gVec(3 * mesh->nVertices());
    MatrixUtils::MatrixIntoColumn(gradient, gVec);
    Vector2 exps = energy->GetExponents();
    double s = 4 - 2 * Hs::get_s(exps.x, exps.y);

    long gradientEndTime = currentTimeMilliseconds();

    Hs::HsMetric hs(energy);

    for (int i = 0; i < 1; i++)
    {
      std::cout << "\nTesting for s = " << s << std::endl;
      long denseStartTime = currentTimeMilliseconds();
      // Dense multiplication
      // Assemble the dense operator
      std::cout << "Multiplying dense" << std::endl;
      Eigen::MatrixXd dense, dense_small;
      dense_small.setZero(mesh->nVertices(), mesh->nVertices());
      dense.setZero(3 * mesh->nVertices(), 3 * mesh->nVertices());
      hs.FillMatrixFracOnly(dense_small, s, mesh, geom);
      MatrixUtils::TripleMatrix(dense_small, dense);
      long denseAssemblyTime = currentTimeMilliseconds();
      // Multiply dense
      Eigen::VectorXd denseRes = dense * gVec;
      long denseEndTime = currentTimeMilliseconds();

      std::cout << "Multiplying BCT" << std::endl;
      // Block cluster tree multiplication
      long bctStartTime = currentTimeMilliseconds();
      BVHNode6D *bvh = energy->GetBVH();

      BlockClusterTree *bct = new BlockClusterTree(mesh, geom, bvh, bh_theta, s);
      bct->PrintData();
      long bctAssemblyTime = currentTimeMilliseconds();
      Eigen::VectorXd bctRes = gVec;
      bct->MultiplyVector3(gVec, bctRes);
      long bctEndTime = currentTimeMilliseconds();

      // Eigen::MatrixXd comp(3 * mesh->nVertices(), 2);
      // comp.col(0) = denseRes;
      // comp.col(1) = bctRes;
      // std::cout << comp << std::endl;

      Eigen::VectorXd diff = denseRes - bctRes;
      double error = 100 * diff.norm() / denseRes.norm();

      std::cout << "Dense multiply norm = " << denseRes.norm() << std::endl;
      std::cout << "Hierarchical multiply norm = " << bctRes.norm() << std::endl;
      std::cout << "Relative error = " << error << " percent" << std::endl;
      std::cout << "Dot product of directions = " << denseRes.dot(bctRes) / (denseRes.norm() * bctRes.norm()) << std::endl;

      long gradientAssembly = gradientEndTime - gradientStartTime;
      long denseAssembly = denseAssemblyTime - denseStartTime;
      long denseMult = denseEndTime - denseAssemblyTime;
      long bctAssembly = bctAssemblyTime - bctStartTime;
      long bctMult = bctEndTime - bctAssemblyTime;

      bct->PrintData();

      std::cout << "Gradient assembly time = " << gradientAssembly << " ms" << std::endl;
      std::cout << "Dense time = " << (denseAssembly + denseMult) << " ms" << std::endl;
      std::cout << "  * Dense matrix assembly = " << denseAssembly << " ms" << std::endl;
      std::cout << "  * Dense multiply = " << denseMult << " ms" << std::endl;
      std::cout << "Hierarchical time = " << (bctAssembly + bctMult) << " ms" << std::endl;
      std::cout << "  * Tree assembly = " << bctAssembly << " ms" << std::endl;
      std::cout << "  * Tree multiply = " << bctMult << " ms" << std::endl;

      delete bct;
      s -= 0.2;
    }
  }

  class VectorInit
  {
  public:
    static void Init(Vector3 &data, BVHNode6D *node)
    {
      data = Vector3{1, 2, 3};
    }
  };

  void MainApp::BenchmarkBH()
  {
    SurfaceEnergy *energy = flow->BaseEnergy();
    Eigen::MatrixXd diff(mesh->nVertices(), 3);

    long totalBVH = 0, totalE = 0, totalG = 0;

    const int nTrials = 100;

    for (int i = 0; i < nTrials; i++)
    {
      diff.setZero();

      long bvhStart = currentTimeMilliseconds();
      energy->Update();
      long eStart = currentTimeMilliseconds();
      double eVal = energy->Value();
      long mid = currentTimeMilliseconds();
      energy->Differential(diff);
      long gEnd = currentTimeMilliseconds();

      long bvhTime = (eStart - bvhStart);
      long eTime = (mid - eStart);
      long gTime = (gEnd - mid);

      totalBVH += bvhTime;
      totalE += eTime;
      totalG += gTime;

      std::cout << i << ": BVH " << bvhTime << " ms, energy " << eTime << " ms, gradient " << gTime << " ms" << std::endl;
    }

    std::cout << "Average over " << nTrials << " runs:" << std::endl;
    std::cout << "BVH construction:    " << ((double)totalBVH / nTrials) << " ms" << std::endl;
    std::cout << "Energy evaluation:   " << ((double)totalE / nTrials) << " ms" << std::endl;
    std::cout << "Gradient evaluation: " << ((double)totalG / nTrials) << " ms" << std::endl;
  }

  void MainApp::AddObstacle(std::string filename, double weight)
  {
    std::unique_ptr<HalfedgeMesh> obstacleMesh;
    std::unique_ptr<VertexPositionGeometry> obstacleGeometry;
    // Load mesh
    std::tie(obstacleMesh, obstacleGeometry) = loadMesh(filename);

    obstacleGeometry->requireVertexDualAreas();
    obstacleGeometry->requireVertexNormals();

    std::string mesh_name = polyscope::guessNiceNameFromPath(filename);
    polyscope::SurfaceMesh *psMesh = polyscope::registerSurfaceMesh(mesh_name, obstacleGeometry->inputVertexPositions,
                                                                    obstacleMesh->getFaceVertexList(), polyscopePermutations(*obstacleMesh));

    double exp = kernel->beta - kernel->alpha;
    StaticObstacle *obstacle = new StaticObstacle(mesh, geom, std::move(obstacleMesh), std::move(obstacleGeometry), exp, bh_theta, weight);
    flow->AddAdditionalEnergy(obstacle);
    std::cout << "Added " << filename << " as obstacle with weight " << weight << std::endl;
  }
} // namespace rsurfaces

// UI parameters
bool run = false;
bool takeScreenshots = false;
uint screenshotNum = 0;
bool uiNormalizeView = false;
int stepLimit;
int numSteps;

void saveScreenshot(uint i)
{
  char buffer[5];
  std::snprintf(buffer, sizeof(buffer), "%04d", screenshotNum);
  std::string fname = "frames/frame" + std::string(buffer) + ".png";
  polyscope::screenshot(fname, false);
  std::cout << "Saved screenshot to " << fname << std::endl;
}

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void customCallback()
{
  using namespace rsurfaces;

  const int INDENT = 10;
  const int ITEM_WIDTH = 160;

  ImGui::Text("Flow control");
  ImGui::BeginGroup();
  ImGui::Indent(INDENT);
  ImGui::PushItemWidth(ITEM_WIDTH);
  ImGui::Checkbox("Run flow", &run);
  ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
  ImGui::Checkbox("Normalize view", &uiNormalizeView);

  ImGui::Checkbox("Take screenshots", &takeScreenshots);
  ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
  if ((takeScreenshots && screenshotNum == 0) || ImGui::Button("Take screenshot", ImVec2{ITEM_WIDTH, 0}))
  {
    saveScreenshot(screenshotNum++);
  }

  ImGui::Text("Iteration limit");
  ImGui::InputInt("", &stepLimit);

  if (uiNormalizeView != MainApp::instance->normalizeView)
  {
    rsurfaces::MainApp::instance->normalizeView = uiNormalizeView;
    rsurfaces::MainApp::instance->updateMeshPositions();
  }
  ImGui::PopItemWidth();
  ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
  if (ImGui::Button("Take 1 step", ImVec2{ITEM_WIDTH, 0}) || run)
  {
    MainApp::instance->TakeFractionalSobolevStep();
    // rsurfaces::MainApp::instance->TakeNaiveStep(0.01);
    MainApp::instance->updateMeshPositions();
    if (takeScreenshots)
    {
      saveScreenshot(screenshotNum++);
    }
    numSteps++;
    if (stepLimit > 0 && numSteps >= stepLimit)
    {
      run = false;
    }
  }
  ImGui::EndGroup();

  ImGui::Text("Accuracy tests");

  ImGui::BeginGroup();
  ImGui::Indent(INDENT);
  if (ImGui::Button("Test LML inv", ImVec2{ITEM_WIDTH, 0}))
  {
    MainApp::instance->TestLML();
  }
  ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
  if (ImGui::Button("Test MV product", ImVec2{ITEM_WIDTH, 0}))
  {
    MainApp::instance->TestMVProduct();
  }

  if (ImGui::Button("Test B-H", ImVec2{ITEM_WIDTH, 0}))
  {
    MainApp::instance->TestBarnesHut();
  }
  ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
  if (ImGui::Button("Benchmark B-H", ImVec2{ITEM_WIDTH, 0}))
  {
    MainApp::instance->BenchmarkBH();
  }

  if (ImGui::Button("Plot face energies", ImVec2{ITEM_WIDTH, 0}))
  {
    MainApp::instance->PlotEnergyPerFace();
  }
  ImGui::SameLine(ITEM_WIDTH, 2 * INDENT);
  if (ImGui::Button("Scale mesh 2x", ImVec2{ITEM_WIDTH, 0}))
  {
    MainApp::instance->Scale2x();
  }

  if (ImGui::Button("Normal deriv", ImVec2{ITEM_WIDTH, 0}))
  {
    MainApp::instance->TestNormalDeriv();
  }
  ImGui::EndGroup();

  ImGui::Text("Remeshing tests");

  ImGui::BeginGroup();
  ImGui::Indent(INDENT);

  // Section for remeshing tests

  if (ImGui::Button("Fix Delaunay"))
  {
    remeshing::fixDelaunay(MainApp::instance->mesh, MainApp::instance->geom);
    MainApp::instance->reregisterMesh();
  }

  if (ImGui::Button("Laplacian smooth"))
  {
    remeshing::smoothByLaplacian(MainApp::instance->mesh, MainApp::instance->geom);
    MainApp::instance->reregisterMesh();
  }

  if (ImGui::Button("Circumcenter smooth"))
  {
    remeshing::smoothByCircumcenter(MainApp::instance->mesh, MainApp::instance->geom);
    MainApp::instance->reregisterMesh();
  }

  if (ImGui::Button("Laplacian optimize"))
  {
    for (int i = 0; i < 1000; i++)
    {
      remeshing::smoothByLaplacian(MainApp::instance->mesh, MainApp::instance->geom);
      remeshing::fixDelaunay(MainApp::instance->mesh, MainApp::instance->geom);
    }
    MainApp::instance->reregisterMesh();
  }

  if (ImGui::Button("Circumcenter optimize"))
  {
    for (int i = 0; i < 1000; i++)
    {
      remeshing::smoothByCircumcenter(MainApp::instance->mesh, MainApp::instance->geom);
      remeshing::fixDelaunay(MainApp::instance->mesh, MainApp::instance->geom);
    }
    MainApp::instance->reregisterMesh();
  }
  ImGui::EndGroup();
}

struct MeshAndEnergy
{
  rsurfaces::TPEKernel *kernel;
  polyscope::SurfaceMesh *psMesh;
  rsurfaces::MeshPtr mesh;
  rsurfaces::GeomPtr geom;
  std::string meshName;
};

MeshAndEnergy initTPEOnMesh(std::string meshFile, double alpha, double beta)
{
  using namespace rsurfaces;

  std::unique_ptr<HalfedgeMesh> u_mesh;
  std::unique_ptr<VertexPositionGeometry> u_geometry;
  // Load mesh
  std::tie(u_mesh, u_geometry) = loadMesh(meshFile);
  std::string mesh_name = polyscope::guessNiceNameFromPath(meshFile);

  // Register the mesh with polyscope
  polyscope::SurfaceMesh *psMesh = polyscope::registerSurfaceMesh(mesh_name,
                                                                  u_geometry->inputVertexPositions, u_mesh->getFaceVertexList(),
                                                                  polyscopePermutations(*u_mesh));

  MeshPtr meshShared = std::move(u_mesh);
  GeomPtr geomShared = std::move(u_geometry);

  geomShared->requireFaceNormals();
  geomShared->requireFaceAreas();
  geomShared->requireVertexNormals();
  geomShared->requireVertexDualAreas();

  TPEKernel *tpe = new rsurfaces::TPEKernel(meshShared, geomShared, alpha, beta);

  return MeshAndEnergy{tpe, psMesh, meshShared, geomShared, mesh_name};
}

rsurfaces::SurfaceFlow *setUpFlow(MeshAndEnergy &m, double theta, rsurfaces::scene::SceneData &scene)
{
  using namespace rsurfaces;

  SurfaceEnergy *energy;

  if (theta <= 0)
  {
    std::cout << "Theta was zero (or negative); using exact all-pairs energy." << std::endl;
    energy = new AllPairsTPEnergy(m.kernel);
  }
  else
  {
    std::cout << "Using Barnes-Hut energy with theta = " << theta << "." << std::endl;
    energy = new BarnesHutTPEnergy6D(m.kernel, theta);
  }

  SurfaceFlow *flow = new SurfaceFlow(energy);
  bool kernelRemoved = false;
  // Set this up here, so that we can aggregate all vertex pins into the same constraint
  Constraints::VertexPinConstraint *c = 0;

  for (scene::ConstraintData &data : scene.constraints)
  {
    switch (data.type)
    {
    case scene::ConstraintType::Barycenter:
      kernelRemoved = true;
      flow->addSimpleConstraint<Constraints::BarycenterConstraint3X>(m.mesh, m.geom);
      break;
    case scene::ConstraintType::TotalArea:
      flow->addSchurConstraint<Constraints::TotalAreaConstraint>(m.mesh, m.geom, data.targetMultiplier, data.numIterations);
      break;
    case scene::ConstraintType::TotalVolume:
      flow->addSchurConstraint<Constraints::TotalVolumeConstraint>(m.mesh, m.geom, data.targetMultiplier, data.numIterations);
      break;
    case scene::ConstraintType::BoundaryPins:
    {
      if (!c)
      {
        c = flow->addSimpleConstraint<Constraints::VertexPinConstraint>(m.mesh, m.geom);
      }
      // Manually add all of the boundary vertex indices as pins
      std::vector<size_t> boundaryInds;
      VertexIndices inds = m.mesh->getVertexIndices();
      for (GCVertex v : m.mesh->vertices())
      {
        if (v.isBoundary())
        {
          boundaryInds.push_back(inds[v]);
        }
      }
      c->pinVertices(m.mesh, m.geom, boundaryInds);
      kernelRemoved = true;
    }
    case scene::ConstraintType::VertexPins:
    {
      if (!c)
      {
        c = flow->addSimpleConstraint<Constraints::VertexPinConstraint>(m.mesh, m.geom);
      }
      // Add the specified vertices as pins
      c->pinVertices(m.mesh, m.geom, scene.vertexPins);
      // Clear the data vector so that we don't add anything twice
      scene.vertexPins.clear();
      kernelRemoved = true;
    }
    break;
    default:
      std::cout << "  * Skipping unrecognized constraint type" << std::endl;
      break;
    }
  }

  if (!kernelRemoved)
  {
    std::cout << "Auto-adding barycenter constraint to eliminate constant kernel of Laplacian" << std::endl;
    flow->addSimpleConstraint<Constraints::BarycenterConstraint3X>(m.mesh, m.geom);
  }

  return flow;
}

rsurfaces::scene::SceneData defaultScene(std::string meshName)
{
  using namespace rsurfaces;
  using namespace rsurfaces::scene;
  SceneData data;
  data.meshName = meshName;
  data.alpha = 6;
  data.beta = 12;
  data.constraints = std::vector<ConstraintData>({ConstraintData{scene::ConstraintType::TotalArea, 1, 0},
                                                  ConstraintData{scene::ConstraintType::TotalVolume, 1, 0}});
  return data;
}

int main(int argc, char **argv)
{
  using namespace rsurfaces;

  // Configure the argument parser
  args::ArgumentParser parser("geometry-central & Polyscope example project");
  args::Positional<std::string> inputFilename(parser, "mesh", "A mesh file.");
  args::ValueFlag<double> thetaFlag(parser, "Theta", "Theta value for Barnes-Hut approximation; 0 means exact.", args::Matcher{'t', "theta"});
  args::ValueFlagList<std::string> obstacleFiles(parser, "obstacles", "Obstacles to add", {'o'});

  int default_threads = omp_get_max_threads();
  std::cout << "OMP autodetected " << default_threads << " threads." << std::endl;
  omp_set_num_threads(default_threads - 2);

  // Parse args
  try
  {
    parser.ParseCLI(argc, argv);
  }
  catch (args::Help)
  {
    std::cout << parser;
    return 0;
  }
  catch (args::ParseError e)
  {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  // Make sure a mesh name was given
  if (!inputFilename)
  {
    std::cerr << "Please specify a mesh file as argument" << std::endl;
    return EXIT_FAILURE;
  }

  double theta = 1;
  if (!thetaFlag)
  {
    std::cout << "Barnes-Hut theta value not specified; defaulting to theta = " << theta << std::endl;
  }
  else
  {
    theta = args::get(thetaFlag);
  }

  // Initialize polyscope
  polyscope::init();
  // Set the callback function
  polyscope::state::userCallback = customCallback;

  // Parse the input file, either as a scene file or as a mesh
  std::string inFile = args::get(inputFilename);
  scene::SceneData data;

  if (endsWith(inFile, ".txt"))
  {
    std::cout << "Parsing " << inFile << " as scene file." << std::endl;
    data = scene::parseScene(inFile);
  }

  else if (endsWith(inFile, ".obj"))
  {
    std::cout << "Parsing " << inFile << " as OBJ mesh file." << std::endl;
    data = defaultScene(inFile);
  }

  MeshAndEnergy m = initTPEOnMesh(data.meshName, 6, 12);
  SurfaceFlow *flow = setUpFlow(m, theta, data);

  MainApp::instance = new MainApp(m.mesh, m.geom, flow, m.psMesh, m.meshName);
  MainApp::instance->bh_theta = theta;
  MainApp::instance->kernel = m.kernel;

  for (scene::ObstacleData &obs : data.obstacles)
  {
    MainApp::instance->AddObstacle(obs.obstacleName, obs.weight);
  }

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
