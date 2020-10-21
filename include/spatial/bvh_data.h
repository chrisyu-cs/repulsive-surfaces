#pragma once

#include "spatial/bvh_types.h"

namespace rsurfaces
{
    inline double nodeRatioBox(Vector3 minCoords, Vector3 maxCoords, double d)
    {
        // Compute diagonal distance from corner to corner
        Vector3 diag = maxCoords - minCoords;
        return diag.norm() / d;
    }

    struct BVHData
    {
        // Basic spatial data
        double totalMass;
        Vector3 centerOfMass;
        Vector3 averageNormal;
        Vector3 minCoords;
        Vector3 maxCoords;
        // Indexing and other metadata
        size_t elementID;
        size_t nodeID;
        BVHNodeType nodeType;
        size_t numNodesInBranch;
        size_t nElements;
        // Indices of children
        size_t child[2];

        bool isAdmissibleFrom(Vector3 atPos, double thresholdTheta) const;
        GCFace getSingleFace(MeshPtr mesh) const;
        MassNormalPoint GetMassNormalPoint() const;
    };
} // namespace rsurfaces