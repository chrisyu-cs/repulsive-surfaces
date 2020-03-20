#pragma once

#include "bvh_types.h"

namespace rsurfaces
{

class BVHNode6D
{
public:
    // Build a BVH of the given points
    BVHNode6D(std::vector<MassNormalPoint> &points, int axis, BVHNode6D *root);
    ~BVHNode6D();

    // Basic spatial data
    double totalMass;
    Vector3 centerOfMass;
    Vector3 averageNormal;
    Vector3 minCoords;
    Vector3 maxCoords;
    // Vector3 averageTangent;
    size_t elementID;
    // Every node knows the root of the tree
    BVHNode6D *bvhRoot;
    // Children
    std::vector<BVHNode6D *> children;
    BVHNodeType nodeType;
    int splitAxis;
    double splitPoint;
    double thresholdTheta;
    size_t nodeID;
    size_t assignIDsRecursively(size_t startID);
    size_t numNodesInBranch;

    void addAllFaces(MeshPtr &mesh, std::vector<GCFace> &faces);

    // Recursively recompute all centers of mass in this tree
    void recomputeCentersOfMass(MeshPtr &mesh, GeomPtr &geom);
    bool isAdmissibleFrom(Vector3 vertPos);
    void printSummary();
    MassNormalPoint GetMassNormalPoint();
    GCFace getSingleFace(MeshPtr &mesh);

private:
    double AxisSplittingPlane(std::vector<MassNormalPoint> &points, int axis);
    void averageDataFromChildren();

    inline double nodeRatio(double d)
    {
        // Compute diagonal distance from corner to corner
        Vector3 diag = maxCoords - minCoords;
        double maxCoord = fmax(diag.x, fmax(diag.y, diag.z));
        return diag.norm() / d;
    }
};

BVHNode6D *Create6DBVHFromMesh(MeshPtr &mesh, GeomPtr &geom);
} // namespace rsurfaces
