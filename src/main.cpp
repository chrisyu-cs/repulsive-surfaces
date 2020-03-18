#include "main.h"

#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "args/args.hxx"
#include "imgui.h"
#include "surface_derivatives.h"

#include "energy/tpe_kernel.h"
#include "energy/all_pairs_tpe.h"
#include "energy/barnes_hut_tpe.h"
#include "helpers.h"

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

  tree = CreateBVHFromMesh(mesh, geom);
  std::cout << tree->numNodesInBranch << " total nodes in BVH" << std::endl;
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
  psMesh->updateVertexPositions(geom->inputVertexPositions);
  polyscope::requestRedraw();
}

void MainApp::PlotL2Gradient()
{
  long start = currentTimeMilliseconds();

  Eigen::MatrixXd d;
  d.setZero(mesh->nVertices(), 3);
  flow->BaseEnergy()->Differential(d);

  std::vector<Vector3> vecs(mesh->nVertices());

  for (size_t i = 0; i < mesh->nVertices(); i++)
  {
    Vector3 v = GetRow(d, i);
    vecs[i] = v;
  }

  psMesh->addVertexVectorQuantity("L2 gradient", vecs);

  long end = currentTimeMilliseconds();

  std::cout << "Plotted gradient in " << (end - start) << " ms" << std::endl;
}
} // namespace rsurfaces

// UI parameters
bool run = false;
bool takeScreenshots = false;
uint screenshotNum = 0;

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

  if (takeScreenshots && screenshotNum == 0)
  {
    saveScreenshot(screenshotNum++);
  }

  if (ImGui::Button("Take step") || run)
  {
    rsurfaces::MainApp::instance->TakeLineSearchStep();
    // rsurfaces::MainApp::instance->TakeNaiveStep(0.01);
    rsurfaces::MainApp::instance->updatePolyscopeMesh();
    if (takeScreenshots) {
      saveScreenshot(screenshotNum++);
    }
  }

  if (ImGui::Button("Plot gradient"))
  {
    rsurfaces::MainApp::instance->PlotL2Gradient();
  }
}

int main(int argc, char **argv)
{

  // Configure the argument parser
  args::ArgumentParser parser("geometry-central & Polyscope example project");
  args::Positional<std::string> inputFilename(parser, "mesh", "A mesh file.");

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

  // Register the mesh with polyscope
  polyscope::SurfaceMesh *psMesh = polyscope::registerSurfaceMesh(
      polyscope::guessNiceNameFromPath(args::get(inputFilename)),
      u_geometry->inputVertexPositions, u_mesh->getFaceVertexList(),
      polyscopePermutations(*u_mesh));

  rsurfaces::MeshPtr meshShared = std::move(u_mesh);
  rsurfaces::GeomPtr geomShared = std::move(u_geometry);

  rsurfaces::TPEKernel *tpe = new rsurfaces::TPEKernel(meshShared, geomShared, 4, 8);
  rsurfaces::AllPairsTPEnergy *energy = new rsurfaces::AllPairsTPEnergy(tpe);
  rsurfaces::SurfaceFlow *flow = new rsurfaces::SurfaceFlow(energy);

  rsurfaces::MainApp::instance = new rsurfaces::MainApp(meshShared, geomShared, flow, psMesh);

  rsurfaces::BarnesHutTPEnergy *bh_energy = new rsurfaces::BarnesHutTPEnergy(tpe, rsurfaces::MainApp::instance->tree);
  Eigen::MatrixXd bh_deriv;
  bh_deriv.setZero(meshShared->nVertices(), 3);
  bh_energy->Differential(bh_deriv);
  std::cout << "BH derivative norm = " << bh_deriv.norm() << std::endl;

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
