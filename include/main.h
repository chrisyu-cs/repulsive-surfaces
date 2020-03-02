#pragma once

#include "rsurface_types.h"
#include "surface_flow.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

namespace rsurfaces
{
class MainApp
{
    public:
    static MainApp* instance;
    MainApp(MeshPtr mesh_, GeomPtr geom_, SurfaceFlow* flow_, polyscope::SurfaceMesh* psMesh_);

    void TakeNaiveStep(double t);
    void updatePolyscopeMesh();

    MeshPtr mesh;
    GeomPtr geom;
    SurfaceFlow* flow;
    polyscope::SurfaceMesh *psMesh;
};
} // namespace rsurfaces
