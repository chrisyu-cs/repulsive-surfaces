#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
class SurfaceFlow
{
public:
    SurfaceFlow(SurfaceEnergy *energy_);
    void StepNaive(double t);

private:
    SurfaceEnergy *energy;
    MeshPtr mesh;
    GeomPtr geom;
};
} // namespace rsurfaces