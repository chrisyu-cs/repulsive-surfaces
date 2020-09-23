#pragma once

#include "rsurface_types.h"
#include "surface_energy.h"

namespace rsurfaces
{

enum class BVHNodeType
{
    Empty,
    Leaf,
    Interior
};

}