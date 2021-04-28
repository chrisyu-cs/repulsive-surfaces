#pragma once

#include "rsurface_types.h"
#include "surface_flow.h"
#include "remeshing/dynamic_remesher.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "scene_file.h"
#include "geometrycentral/surface/meshio.h"

#include "energy/squared_error.h"

#include <mkl.h>
#include "optimized_bct.h"
#include "energy/willmore_energy.h"
#include "energy/tpe_multipole_0.h"
#include "energy/tpe_barnes_hut_0.h"
#include "implicit/simple_surfaces.h"
#include "marchingcubes/CIsoSurface.h"

#define EIGEN_NO_DEBUG

namespace rsurfaces
{
    struct PriorityVertex
    {
        GCVertex vertex;
        double priority;
        Vector3 position;
    };

    class MainApp
    {
    public:
        static MainApp *instance;
        MainApp(MeshPtr mesh_, GeomPtr geom_, SurfaceFlow *flow_, polyscope::SurfaceMesh *psMesh_, std::string meshName_);

        void CreateAndDestroyBVH();
        void TestWillmore();
        void TestMultiply();
        void TestUpdate();
        void TestObstacle0();
        void TestBarnesHut0();
        void PlotGradients();
        void Scale2x();
        void TestNormalDeriv();
        void MeshImplicitSurface(ImplicitSurface *surface);

        void GetFalloffWindow(GCVertex v, double radius, std::vector<PriorityVertex> &verts);
        void HandlePicking();

        void TakeOptimizationStep(bool remeshAfter, bool showAreaRatios);
        void AddObstacle(std::string filename, double weight, bool recenter, bool asPointCloud);
        void AddPotential(scene::PotentialType pType, double weight);
        void AddImplicitBarrier(scene::ImplicitBarrierData &implicitBarrier);

        MeshPtr mesh;
        GeomPtr geom;
        GeomPtr geomOrig;
        UVDataPtr uvs;
        SurfaceFlow *flow;
        TPEKernel *kernel;
        TPEnergyAllPairs *referenceEnergy;
        
        polyscope::SurfaceMesh *psMesh;
        std::vector<polyscope::SurfaceMesh *> obstacles;
        std::string meshName;
        int stepLimit;
        int realTimeLimit;
        GradientMethod methodChoice;

        inline void reregisterMesh()
        {
            psMesh = polyscope::registerSurfaceMesh(meshName, geom->inputVertexPositions, mesh->getFaceVertexList(), polyscopePermutations(*mesh));
        }

        void updateMeshPositions();

        BVHNode6D *vertBVH;
        bool normalizeView;
        double bh_theta;
        remeshing::DynamicRemesher remesher;
        int numSteps;
        bool logPerformance;
        long timeSpentSoFar;
        scene::SceneData sceneData;
        bool exitWhenDone;
        double totalObstacleVolume;

    private:
        int implicitCount = 0;
        GCVertex pickedVertex;
        std::vector<PriorityVertex> dragVertices;

        double pickDepth;
        void logPerformanceLine();
        bool pickNearbyVertex(GCVertex &out);
        SquaredError *vertexPotential;
        bool ctrlMouseDown;
        Vector3 initialPickedPosition;
        bool hasPickedVertex;
    };
} // namespace rsurfaces
