#pragma once

#include "energy/tpe_kernel.h"
#include "spatial/bvh_6d.h"

namespace rsurfaces
{

class BarnesHutTPEnergy6D : public SurfaceEnergy
{
public:
    BarnesHutTPEnergy6D(TPEKernel *kernel_, BVHNode6D *root_);
    virtual double Value();
    virtual void Differential(Eigen::MatrixXd &output);
    virtual MeshPtr GetMesh();
    virtual GeomPtr GetGeom();

private:
    TPEKernel *kernel;
    BVHNode6D *root;
    double computeEnergyOfFace(GCFace face, BVHNode6D *bvhRoot);
};

} // namespace rsurfaces