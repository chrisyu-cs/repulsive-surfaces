#include "main.h"

#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"


#include "args/args.hxx"
#include "imgui.h"
#include "surface_derivatives.h"
#include "tpe_energy_surface.h"

#include "all_pairs_tpe.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace rsurfaces {

    MainApp* MainApp::instance = 0;

    MainApp::MainApp(MeshPtr mesh_, GeomPtr geom_, SurfaceFlow* flow_, polyscope::SurfaceMesh* psMesh_) {
      mesh = mesh_;
      geom = geom_;
      flow = flow_;
      psMesh = psMesh_;
    }

    void MainApp::TakeNaiveStep(double t) {
      std::cout << geom->inputVertexPositions[mesh->vertex(0)] << " -> ";
      flow->StepNaive(t);
      std::cout << geom->inputVertexPositions[mesh->vertex(0)] << std::endl;
    }

    void MainApp::updatePolyscopeMesh() {
      psMesh->updateVertexPositions(geom->inputVertexPositions);
      polyscope::requestRedraw();
    }
}

// Some algorithm parameters
float param1 = 42.0;
bool run = false;

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback()
{
  ImGui::Checkbox("Run flow", &run);

  if (ImGui::Button("do work") || run)
  {
    rsurfaces::MainApp::instance->TakeNaiveStep(0.1);
    rsurfaces::MainApp::instance->updatePolyscopeMesh();
  }

  ImGui::SliderFloat("param", &param1, 0., 100.);
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
  polyscope::SurfaceMesh* psMesh = polyscope::registerSurfaceMesh(
      polyscope::guessNiceNameFromPath(args::get(inputFilename)),
      u_geometry->inputVertexPositions, u_mesh->getFaceVertexList(),
      polyscopePermutations(*u_mesh));

  rsurfaces::MeshPtr meshShared = std::move(u_mesh);
  rsurfaces::GeomPtr geomShared = std::move(u_geometry);

  rsurfaces::TPEKernel* tpe = new rsurfaces::TPEKernel(meshShared, geomShared, 3, 6);
  rsurfaces::AllPairsTPEnergy* energy = new rsurfaces::AllPairsTPEnergy(tpe);
  rsurfaces::SurfaceFlow* flow = new rsurfaces::SurfaceFlow(energy);

  rsurfaces::MainApp::instance = new rsurfaces::MainApp(meshShared, geomShared, flow, psMesh);

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
