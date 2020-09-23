#pragma once

#include "energy/tpe_kernel.h"
#include "spatial/bvh_6d.h"
#include "spatial/bvh_flattened.h"

namespace rsurfaces
{

    // Evaluates energy and differential using Barnes-Hut with a BVH.
    class BarnesHutNewtonian : public SurfaceEnergy
    {
    public:
        BarnesHutNewtonian(TPEKernel *kernel_, double theta_);
        ~BarnesHutNewtonian();
        virtual double Value();
        virtual void Differential(Eigen::MatrixXd &output);
        virtual void Update();
        virtual MeshPtr GetMesh();
        virtual GeomPtr GetGeom();
        virtual Vector2 GetExponents();
        virtual BVHNode6D *GetBVH();
        virtual double GetTheta();

    private:
        TPEKernel *kernel;
        BVHNode6D *root;
        BVHFlattened *flattened;

        double theta;
        double computeEnergyOfFace(GCFace face, BVHNode6D *bvhRoot);
        void accumulateForce(Eigen::MatrixXd &gradients, BVHNode6D *node, GCFace face1,
                             surface::VertexData<size_t> &indices);
    };

} // namespace rsurfaces