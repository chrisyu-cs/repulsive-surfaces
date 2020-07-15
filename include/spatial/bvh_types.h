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


struct MassNormalPoint
{
    double mass;
    Vector3 normal;
    Vector3 point;
    Vector3 minCoords;
    Vector3 maxCoords;
    size_t elementID;
};

enum class BVHNodeType
{
    Empty,
    Leaf,
    Interior
};

}