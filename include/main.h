#pragma once

#include "rsurface_types.h"
#include "surface_flow.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "energy/barnes_hut_tpe_6d.h"
#include "scene_file.h"

namespace rsurfaces
{
    class MainApp
    {
    public:
        static MainApp *instance;
        MainApp(MeshPtr mesh_, GeomPtr geom_, SurfaceFlow *flow_, polyscope::SurfaceMesh *psMesh_, std::string meshName_);

        void TestLML();
        void TestMVProduct();
        void BenchmarkBH();
        void TestBarnesHut();
        void PlotEnergyPerFace();
        void Scale2x();
        void TestNormalDeriv();

        void TakeNaiveStep(double t);
        void TakeFractionalSobolevStep();
        void AddObstacle(std::string filename, double weight);
        void AddPotential(scene::PotentialType pType, double weight);

        MeshPtr mesh;
        GeomPtr geom;
        SurfaceFlow *flow;
        TPEKernel *kernel;
        polyscope::SurfaceMesh *psMesh;
        std::vector<polyscope::SurfaceMesh *> obstacles;
        std::string meshName;

        inline void reregisterMesh()
        {
            psMesh = polyscope::registerSurfaceMesh(meshName, geom->inputVertexPositions, mesh->getFaceVertexList());
        }
        
        void updateMeshPositions();

        BVHNode6D *vertBVH;
        bool normalizeView;
        double bh_theta;
    };
} // namespace rsurfaces
