#pragma once

#include "rsurface_types.h"
#include <vector>
#include <unordered_set>
#include <chrono>

namespace rsurfaces
{

// Find the set of vertices that are on the boundary of either
// triangle, without duplicates
inline void GetVerticesWithoutDuplicates(GCFace f1, GCFace f2, std::vector<GCVertex> &verts)
{
    std::unordered_set<GCVertex> vertInds;
    for (GCVertex v : f1.adjacentVertices())
    {
        if (vertInds.count(v) == 0)
        {
            verts.push_back(v);
            vertInds.insert(v);
        }
    }
    for (GCVertex v : f2.adjacentVertices())
    {
        if (vertInds.count(v) == 0)
        {
            verts.push_back(v);
            vertInds.insert(v);
        }
    }
}

inline Vector3 GetRow(Eigen::MatrixXd &A, int i)
{
    return Vector3{A(i, 0), A(i, 1), A(i, 2)};
}

inline long currentTimeMilliseconds()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

} // namespace rsurfaces
