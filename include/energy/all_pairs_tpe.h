#pragma once

#include "energy/tpe_kernel.h"

namespace rsurfaces
{

class AllPairsTPEnergy : public SurfaceEnergy
{
public:
    AllPairsTPEnergy(TPEKernel *kernel_);
    virtual double Value();
    virtual void Differential(Eigen::MatrixXd &output);
    virtual MeshPtr GetMesh();
    virtual GeomPtr GetGeom();

private:
    TPEKernel *kernel;
};

} // namespace rsurfaces