#pragma once

#include "rsurface_types.h"
#include <vector>
#include <unordered_set>

namespace rsurfaces
{

// Find the set of vertices that are on the boundary of either
// triangle, without duplicates
inline void GetVerticesWithoutDuplicates(GCFace f1, GCFace f2, std::vector<GCVertex> &verts)
{
    std::unordered_set<size_t> vertInds;
    for (GCVertex v : f1.adjacentVertices())
    {
        if (vertInds.count(v.getIndex()) == 0)
        {
            verts.push_back(v);
            vertInds.insert(v.getIndex());
        }
    }
    for (GCVertex v : f2.adjacentVertices())
    {
        if (vertInds.count(v.getIndex()) == 0)
        {
            verts.push_back(v);
            vertInds.insert(v.getIndex());
        }
    }
}

inline Vector3 GetRow(Eigen::MatrixXd &A, int i) {
    return Vector3{A(i, 0), A(i, 1), A(i, 2)};
}

} // namespace rsurfaces
