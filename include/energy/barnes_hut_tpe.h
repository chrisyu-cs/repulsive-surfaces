#pragma once

#include "energy/tpe_energy_surface.h"
#include "spatial/bvh_3d.h"

namespace rsurfaces
{

class BarnesHutTPEnergy : public SurfaceEnergy
{
public:
    BarnesHutTPEnergy(TPEKernel *kernel_, BVHNode3D *root_);
    virtual double Value();
    virtual void Differential(Eigen::MatrixXd &output);
    virtual MeshPtr GetMesh();
    virtual GeomPtr GetGeom();

private:
    TPEKernel *kernel;
    BVHNode3D *root;
    double computeEnergyOfFace(GCFace face, BVHNode3D *bvhRoot);
    void addVOfPair(GCFace face, BVHNode3D *bvhRoot, Eigen::VectorXd &V);
};

} // namespace rsurfaces