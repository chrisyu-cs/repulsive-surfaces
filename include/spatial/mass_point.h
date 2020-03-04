#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{

struct MassPoint
{
    double mass;
    Vector3 point;
    size_t elementID;
};

} // namespace rsurfaces