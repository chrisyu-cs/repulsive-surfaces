#include "main.h"

#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "args/args.hxx"
#include "imgui.h"
#include "surface_derivatives.h"

#include "energy/tpe_kernel.h"
#include "energy/all_pairs_tpe.h"
#include "helpers.h"
#include <memory>

#include <Eigen/Sparse>

#include "sobolev/constraints.h"
#include "sobolev/hs.h"
#include "sobolev/h1.h"
#include "spatial/convolution.h"
#include "spatial/convolution_kernel.h"

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

  void MainApp::TakeLineSearchStep()
  {
    flow->StepLineSearch();
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

  void PlotMatrix(Eigen::MatrixXd &mat, polyscope::SurfaceMesh *psMesh, std::string name) {
    std::vector<Vector3> vecs;
    for (int i = 0; i < mat.rows(); i++) {
      Vector3 row_i = GetRow(mat, i);
      vecs.push_back(row_i);
    }
    psMesh->addVertexVectorQuantity(name, vecs);
  }

  void PlotVector(Eigen::VectorXd &vec, int nVerts, polyscope::SurfaceMesh *psMesh, std::string name) {
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

  void MainApp::TestLML() {
    SurfaceEnergy *energy = flow->BaseEnergy();
    Vector2 exps = energy->GetExponents();
    energy->Update();

    Eigen::MatrixXd gradient(mesh->nVertices(), 3);
    energy->Differential(gradient);
    Eigen::MatrixXd result = gradient;

    Hs::ProjectViaSparse(gradient, result, exps.x, exps.y, mesh, geom);

    std::cout << result << std::endl;
    PlotMatrix(result, psMesh, "LML approx");
  }

  void MainApp::TestConvolution()
  {
    if (vertBVH)
    {
      delete vertBVH;
    }

    VertexIndices inds = mesh->getVertexIndices();
    vertBVH = Create6DBVHFromMeshVerts(mesh, geom, 0.25);

    int nVerts = mesh->nVertices();
    SurfaceEnergy *energy = flow->BaseEnergy();

    energy->Update();

    Eigen::MatrixXd gradient, gradient_conv, gradient_conv2, gradient_proj;
    gradient.setZero(nVerts, 3);
    gradient_conv.setZero(nVerts, 3);
    gradient_conv2.setZero(nVerts, 3);
    gradient_proj.setZero(nVerts, 3);
    energy->Differential(gradient);
    Vector2 exps = energy->GetExponents();
    double s = Hs::get_s(exps.x, exps.y);

    Eigen::MatrixXd sxs;
    sxs.setZero(nVerts, 6);

    std::cout << "s = " << s << std::endl;

    Hs::ProjectGradient(gradient, gradient_proj, exps.x, exps.y, mesh, geom);

    RieszKernel kernel(2. - s);
    FixBarycenter(mesh, geom, inds, gradient);
    ConvolveExact(mesh, geom, kernel, gradient, gradient_conv);
    ConvolveExact(mesh, geom, kernel, gradient_conv, gradient_conv2);
    FixBarycenter(mesh, geom, inds, gradient_conv2);

    sxs.block(0, 0, nVerts, 3) = gradient_proj;
    sxs.block(0, 3, nVerts, 3) = gradient_conv2;

    std::vector<Vector3> vecs_conv, vecs_proj, vecs_orig;

    for (int i = 0; i < nVerts; i++)
    {
      vecs_conv.push_back(GetRow(gradient_conv2, i));
      vecs_proj.push_back(GetRow(gradient_proj, i));
      vecs_orig.push_back(GetRow(gradient, i));
    }

    Vector3 center{0, 0, 0};
    Vector3 centerProj = center;
    Vector3 centerConv = center;
    double sumArea = 0;

    std::cout << "Norm orig = " << gradient.norm() << std::endl;
    std::cout << "Norm proj = " << gradient_proj.norm() << std::endl;
    std::cout << "Norm conv = " << gradient_conv2.norm() << std::endl;

    for (GCVertex v : mesh->vertices())
    {
      center += GetRow(gradient, inds[v]) * geom->vertexDualAreas[v];
      centerProj += GetRow(gradient_proj, inds[v]) * geom->vertexDualAreas[v];
      centerConv += GetRow(gradient_conv2, inds[v]) * geom->vertexDualAreas[v];
      sumArea += geom->vertexDualAreas[v];
    }

    center /= sumArea;
    centerProj /= sumArea;
    centerConv /= sumArea;

    std::cout << "Orig barycenter = " << center << std::endl;
    std::cout << "Proj barycenter = " << centerProj << std::endl;
    std::cout << "Conv barycenter = " << centerConv << std::endl;

    psMesh->addVertexVectorQuantity("original", vecs_orig);
    psMesh->addVertexVectorQuantity("convolved", vecs_conv);
    psMesh->addVertexVectorQuantity("projected", vecs_proj);

    // std::cout << sxs << std::endl;
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

  if (ImGui::Button("Take screenshot")) {
    saveScreenshot(screenshotNum++);
  }

  if (ImGui::Button("Take step") || run)
  {
    rsurfaces::MainApp::instance->TakeLineSearchStep();
    // rsurfaces::MainApp::instance->TakeNaiveStep(0.01);
    rsurfaces::MainApp::instance->updatePolyscopeMesh();
    if (takeScreenshots)
    {
      saveScreenshot(screenshotNum++);
    }
    numSteps++;
    if (stepLimit > 0 && numSteps >= stepLimit) {
      run = false;
    }
  }

  if (ImGui::Button("Test convolution"))
  {
    rsurfaces::MainApp::instance->TestConvolution();
  }

  if (ImGui::Button("Test LML inverse"))
  {
    rsurfaces::MainApp::instance->TestLML();
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
  MainApp::instance = new MainApp(meshShared, geomShared, flow, psMesh);

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
