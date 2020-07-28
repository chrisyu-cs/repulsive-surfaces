#pragma once

#include "energy/tpe_kernel.h"
#include "spatial/bvh_6d.h"

namespace rsurfaces
{

class BarnesHutTPEnergy6D : public SurfaceEnergy
{
public:
    BarnesHutTPEnergy6D(TPEKernel *kernel_, double theta_);
    virtual double Value();
    virtual void Differential(Eigen::MatrixXd &output);
    virtual void Update();
    virtual MeshPtr GetMesh();
    virtual GeomPtr GetGeom();
    virtual Vector2 GetExponents();
    virtual BVHNode6D* GetBVH();

private:
    TPEKernel *kernel;
    BVHNode6D *root;
    double theta;
    double computeEnergyOfFace(GCFace face, BVHNode6D *bvhRoot);
    void accumulateTPEGradient(Eigen::MatrixXd &gradients, BVHNode6D *node, GCFace face1,
                               surface::VertexData<size_t> indices);
};

} // namespace rsurfaces