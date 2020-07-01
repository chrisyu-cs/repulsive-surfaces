#pragma once

#include "rsurface_types.h"
#include "matrix_utils.h"

namespace rsurfaces
{
    namespace Constraints
    {
        void addBarycenterTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
    } // namespace Constraints

    namespace H1
    {
        void getTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom);
    }
} // namespace rsurfaces
