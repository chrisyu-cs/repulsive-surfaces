#pragma once

#include "energy/tpe_kernel.h"
#include "spatial/bvh_6d.h"

namespace rsurfaces
{

    // Evaluates energy and differential naively with all pairs, but
    // provides a BVH when GetBVH() is called. For testing purposes.
    class AllPairsWithBVH : public SurfaceEnergy
    {
    public:
        AllPairsWithBVH(TPEKernel *kernel_, double theta_);
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
        double theta;
    };

} // namespace rsurfaces