#pragma once

#include "rsurface_types.h"
#include "energy/tpe_kernel.h"
#include "spatial/bvh_6d.h"

namespace rsurfaces
{
    class CoulombEnergy : public SurfaceEnergy
    {
    public:
        CoulombEnergy(TPEKernel *kernel_, double theta_);
        ~CoulombEnergy();
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
        double energyAtVertex(BVHNode6D *node, GCVertex v);
        Vector3 gradientAtVertex(BVHNode6D *node, GCVertex v);
    };
} // namespace rsurfaces