#pragma once

#include "tpe_energy_surface.h"

namespace rsurfaces
{

class AllPairsTPEnergy : public SurfaceEnergy
{
public:
    AllPairsTPEnergy(TPEKernel *kernel_);
    virtual double Value();
    virtual void Differential(Eigen::MatrixXd &output);

private:
    TPEKernel *kernel;
};

} // namespace rsurfaces