#pragma once

#include "rsurface_types.h"
#include "surface_flow.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "energy/barnes_hut_tpe_6d.h"

namespace rsurfaces
{
class MainApp
{
    public:
    static MainApp* instance;
    MainApp(MeshPtr mesh_, GeomPtr geom_, SurfaceFlow* flow_, polyscope::SurfaceMesh* psMesh_);

    void updatePolyscopeMesh();
    void PlotL2Gradient();
    void TestLML();
    void TestMVProduct();
    void TestPercolation();
    void TestBarnesHut();

    void TakeNaiveStep(double t);
    void TakeFractionalSobolevStep();

    MeshPtr mesh;
    GeomPtr geom;
    SurfaceFlow* flow;
    polyscope::SurfaceMesh *psMesh;
    BVHNode6D *vertBVH;
    bool normalizeView;
    double bh_theta;
};
} // namespace rsurfaces
