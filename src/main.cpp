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
#include "helpers.h"
#include <memory>

#include <Eigen/Sparse>
#include <omp.h>

#include "sobolev/all_constraints.h"
#include "sobolev/hs.h"
#include "sobolev/h1.h"
#include "spatial/convolution.h"
#include "spatial/convolution_kernel.h"
#include "block_cluster_tree.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace rsurfaces
{

  MainApp *MainApp::instance = 0;

  MainApp::MainApp(MeshPtr mesh_, GeomPtr geom_, SurfaceFlow *flow_, polyscope::SurfaceMesh *psMesh_)
  {
    mesh = mesh_;
    geom = geom_;
    flow = flow_;
    psMesh = psMesh_;
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

  void MainApp::updatePolyscopeMesh()
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

  void MainApp::PlotL2Gradient()
  {
    long start = currentTimeMilliseconds();
    flow->BaseEnergy()->Update();

    Eigen::MatrixXd d;
    d.setZero(mesh->nVertices(), 3);
    Eigen::MatrixXd h1 = d;
    Eigen::MatrixXd hs = d;
    flow->BaseEnergy()->Differential(d);

    PlotMatrix(d, psMesh, "L2 gradient");

    Vector2 ab = flow->BaseEnergy()->GetExponents();
    H1::ProjectGradient(d, h1, mesh, geom);
    Hs::ProjectGradient(d, hs, ab.x, ab.y, mesh, geom);
    PlotMatrix(h1, psMesh, "H1 gradient");
    PlotMatrix(hs, psMesh, "Hs gradient");
  }

  void MainApp::TestLML()
  {
    SurfaceEnergy *energy = flow->BaseEnergy();
    Vector2 exps = energy->GetExponents();
    energy->Update();

    Eigen::MatrixXd gradient(mesh->nVertices(), 3);
    energy->Differential(gradient);
    Eigen::MatrixXd result = gradient;

    BlockClusterTree *bct = 0;
    Hs::SparseFactorization factor;
    Hs::ProjectViaSparseMat(gradient, result, energy, bct, factor);
    delete bct;

    std::cout << result << std::endl;
    PlotMatrix(result, psMesh, "LML approx");
  }

  void MainApp::TestBarnesHut()
  {
    TPEKernel *tpe = new rsurfaces::TPEKernel(mesh, geom, 6, 12);
    SurfaceEnergy *energy_ap, *energy_bh, *energy_xd;
    energy_ap = new AllPairsTPEnergy(tpe);
    energy_bh = new BarnesHutTPEnergy6D(tpe, bh_theta);
    energy_xd = new BarnesHutTPEnergyXDiff(tpe, bh_theta);

    energy_ap->Update();
    energy_bh->Update();
    energy_xd->Update();

    Eigen::MatrixXd grad_ap(mesh->nVertices(), 3);
    Eigen::MatrixXd grad_bh(mesh->nVertices(), 3);
    Eigen::MatrixXd grad_xd(mesh->nVertices(), 3);
    grad_ap.setZero();
    grad_bh.setZero();
    grad_xd.setZero();

    long start_ape = currentTimeMilliseconds();
    double value_ap = energy_ap->Value();
    long end_ape = currentTimeMilliseconds();

    long start_bhe = currentTimeMilliseconds();
    double value_bh = energy_bh->Value();
    long end_bhe = currentTimeMilliseconds();

    long start_xde = currentTimeMilliseconds();
    double value_xd = energy_xd->Value();
    long end_xde = currentTimeMilliseconds();

    double val_error = fabs(value_ap - value_bh) / value_ap;

    std::cout << "\n=====   Energy   =====" << std::endl;
    std::cout << "All-pairs energy value  = " << value_ap << std::endl;
    std::cout << "Barnes-Hut energy value = " << value_bh << std::endl;
    std::cout << "BH exact diff value     = " << value_xd << std::endl;
    std::cout << "Relative error     = " << val_error * 100 << " percent" << std::endl;
    std::cout << "All-pairs time     = " << (end_ape - start_ape) << " ms" << std::endl;
    std::cout << "Barnes-Hut time    = " << (end_bhe - start_bhe) << " ms" << std::endl;
    std::cout << "BH exact diff time = " << (end_xde - start_xde) << " ms" << std::endl;

    long start_apg = currentTimeMilliseconds();
    energy_ap->Differential(grad_ap);
    long end_apg = currentTimeMilliseconds();

    long start_bhg = currentTimeMilliseconds();
    energy_bh->Differential(grad_bh);
    long end_bhg = currentTimeMilliseconds();

    long start_xdg = currentTimeMilliseconds();
    energy_xd->Differential(grad_xd);
    long end_xdg = currentTimeMilliseconds();

    double grad_error = (grad_ap - grad_bh).norm() / grad_ap.norm();
    double xd_error = (grad_ap - grad_xd).norm() / grad_ap.norm();

    std::cout << "\n=====  Gradient  =====" << std::endl;
    std::cout << "All-pairs gradient norm      = " << grad_ap.norm() << std::endl;
    std::cout << "Barnes-Hut gradient norm     = " << grad_bh.norm() << std::endl;
    std::cout << "BH exact diff gradient norm  = " << grad_xd.norm() << std::endl;
    std::cout << "Barnes-Hut relative error    = " << grad_error * 100 << " percent" << std::endl;
    std::cout << "BH exact diff relative error = " << xd_error * 100 << " percent" << std::endl;
    std::cout << "All-pairs time     = " << (end_apg - start_apg) << " ms" << std::endl;
    std::cout << "Barnes-Hut time    = " << (end_bhg - start_bhg) << " ms" << std::endl;
    std::cout << "BH exact diff time = " << (end_xdg - start_xdg) << " ms" << std::endl;

    delete energy_ap;
    delete energy_bh;
    delete energy_xd;
    delete tpe;
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
      Hs::FillMatrixFracOnly(dense_small, s, mesh, geom);
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

  void MainApp::TestPercolation()
  {
    SurfaceEnergy *energy = flow->BaseEnergy();
    energy->Update();
    BVHNode6D *root = energy->GetBVH();

    DataTreeContainer<Vector3> *dtree = root->CreateDataTree<Vector3, VectorInit>();
    std::cout << dtree->tree->data << std::endl;
    delete dtree;
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
void myCallback()
{
  ImGui::Checkbox("Run flow", &run);

  ImGui::Checkbox("Take screenshots", &takeScreenshots);

  ImGui::Checkbox("Normalize view", &uiNormalizeView);

  ImGui::InputInt("Step limit", &stepLimit);

  if (uiNormalizeView != rsurfaces::MainApp::instance->normalizeView)
  {
    rsurfaces::MainApp::instance->normalizeView = uiNormalizeView;
    rsurfaces::MainApp::instance->updatePolyscopeMesh();
  }

  if (takeScreenshots && screenshotNum == 0)
  {
    saveScreenshot(screenshotNum++);
  }

  if (ImGui::Button("Take screenshot"))
  {
    saveScreenshot(screenshotNum++);
  }

  if (ImGui::Button("Take step") || run)
  {
    rsurfaces::MainApp::instance->TakeFractionalSobolevStep();
    // rsurfaces::MainApp::instance->TakeNaiveStep(0.01);
    rsurfaces::MainApp::instance->updatePolyscopeMesh();
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

  if (ImGui::Button("Test LML inverse"))
  {
    rsurfaces::MainApp::instance->TestLML();
  }

  if (ImGui::Button("Test percolation"))
  {

    rsurfaces::MainApp::instance->TestPercolation();
  }

  if (ImGui::Button("Test mat-vec product"))
  {
    rsurfaces::MainApp::instance->TestMVProduct();
  }

  if (ImGui::Button("Test Barnes-Hut"))
  {
    rsurfaces::MainApp::instance->TestBarnesHut();
  }

  if (ImGui::Button("Plot gradient"))
  {
    rsurfaces::MainApp::instance->PlotL2Gradient();
  }
}

void testBarnesHut(rsurfaces::TPEKernel *tpe, rsurfaces::MeshPtr mesh, rsurfaces::GeomPtr geom, rsurfaces::BVHNode6D *bvh)
{
  using namespace rsurfaces;

  std::unique_ptr<AllPairsTPEnergy> exact_energy = std::unique_ptr<AllPairsTPEnergy>(new AllPairsTPEnergy(tpe));
  std::unique_ptr<BarnesHutTPEnergy6D> bh_energy = std::unique_ptr<BarnesHutTPEnergy6D>(new BarnesHutTPEnergy6D(tpe, 0.25));
  Eigen::MatrixXd bh_deriv, exact_deriv;

  bh_deriv.setZero(mesh->nVertices(), 3);
  exact_deriv.setZero(mesh->nVertices(), 3);

  bh_energy->Differential(bh_deriv);
  exact_energy->Differential(exact_deriv);

  std::cout << "BH derivative norm    = " << bh_deriv.norm() << std::endl;
  std::cout << "Exact derivative norm = " << exact_deriv.norm() << std::endl;

  std::cout << "BH first three rows:\n"
            << bh_deriv.block(0, 0, 9, 3) << std::endl;
  std::cout << "Exact first three rows:\n"
            << exact_deriv.block(0, 0, 9, 3) << std::endl;

  double bh_value = bh_energy->Value();
  double exact_value = exact_energy->Value();

  std::cout << "BH energy    = " << bh_value << std::endl;
  std::cout << "Exact energy = " << exact_value << std::endl;

  Eigen::VectorXd bh_vec(3 * mesh->nVertices());
  Eigen::VectorXd exact_vec(3 * mesh->nVertices());

  for (size_t i = 0; i < mesh->nVertices(); i++)
  {
    bh_vec(3 * i + 0) = bh_deriv(i, 0);
    bh_vec(3 * i + 1) = bh_deriv(i, 1);
    bh_vec(3 * i + 2) = bh_deriv(i, 2);

    exact_vec(3 * i + 0) = exact_deriv(i, 0);
    exact_vec(3 * i + 1) = exact_deriv(i, 1);
    exact_vec(3 * i + 2) = exact_deriv(i, 2);
  }

  bh_vec = bh_vec / bh_vec.norm();
  exact_vec = exact_vec / exact_vec.norm();
  double dir_dot = bh_vec.dot(exact_vec);
  std::cout << "Dot product of directions = " << dir_dot << std::endl;
}

int main(int argc, char **argv)
{
  using namespace rsurfaces;

  // Configure the argument parser
  args::ArgumentParser parser("geometry-central & Polyscope example project");
  args::Positional<std::string> inputFilename(parser, "mesh", "A mesh file.");
  args::ValueFlag<double> thetaFlag(parser, "Theta", "Theta value for Barnes-Hut approximation; 0 means exact.", args::Matcher{'t', "theta"});

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

  double theta = 0.5;
  if (!thetaFlag)
  {
    std::cout << "Barnes-Hut theta value not specified; defaulting to theta = 0.5." << std::endl;
  }
  else
  {
    theta = args::get(thetaFlag);
  }

  // Initialize polyscope
  polyscope::init();

  // Set the callback function
  polyscope::state::userCallback = myCallback;

  std::unique_ptr<HalfedgeMesh> u_mesh;
  std::unique_ptr<VertexPositionGeometry> u_geometry;
  // Load mesh
  std::tie(u_mesh, u_geometry) = loadMesh(args::get(inputFilename));
  u_geometry->requireVertexPositions();
  u_geometry->requireFaceNormals();

  std::string mesh_name = polyscope::guessNiceNameFromPath(args::get(inputFilename));

  // Register the mesh with polyscope
  polyscope::SurfaceMesh *psMesh = polyscope::registerSurfaceMesh(mesh_name,
                                                                  u_geometry->inputVertexPositions, u_mesh->getFaceVertexList(),
                                                                  polyscopePermutations(*u_mesh));

  MeshPtr meshShared = std::move(u_mesh);
  GeomPtr geomShared = std::move(u_geometry);

  geomShared->requireVertexDualAreas();
  geomShared->requireVertexNormals();

  TPEKernel *tpe = new rsurfaces::TPEKernel(meshShared, geomShared, 6, 12);

  SurfaceEnergy *energy;

  if (theta <= 0)
  {
    std::cout << "Theta was zero (or negative); using exact all-pairs energy." << std::endl;
    energy = new AllPairsTPEnergy(tpe);
  }
  else
  {
    std::cout << "Using Barnes-Hut energy with theta = " << theta << "." << std::endl;
    energy = new BarnesHutTPEnergy6D(tpe, theta);
  }

  SurfaceFlow *flow = new SurfaceFlow(energy);
  flow->addConstraint<Constraints::TotalVolumeConstraint>(meshShared, geomShared);
  flow->addConstraint<Constraints::TotalAreaConstraint>(meshShared, geomShared);
  MainApp::instance = new MainApp(meshShared, geomShared, flow, psMesh);
  MainApp::instance->bh_theta = theta;

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
