#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
class SurfaceFlow
{
public:
    SurfaceFlow(SurfaceEnergy *energy_);

private:
    SurfaceEnergy *energy;
};
} // namespace rsurfaces