#pragma once

#include "bvh_types.h"
#include "data_tree.h"

namespace rsurfaces
{

    class BVHNode6D
    {
    public:
        // Build a BVH of the given points
        BVHNode6D(std::vector<MassNormalPoint> &points, int axis, BVHNode6D *root, double theta);
        ~BVHNode6D();

        // Basic spatial data
        double totalMass;
        Vector3 centerOfMass;
        Vector3 averageNormal;
        Vector3 minCoords;
        Vector3 maxCoords;
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

        // Store the list of all indices of elements in this cluster
        std::vector<size_t> clusterIndices;

        void addAllFaces(MeshPtr &mesh, std::vector<GCFace> &faces);

        // Recursively recompute all centers of mass in this tree
        void recomputeCentersOfMass(MeshPtr &mesh, GeomPtr &geom);
        bool isAdmissibleFrom(Vector3 vertPos);
        void printSummary();
        MassNormalPoint GetMassNormalPoint();
        GCFace getSingleFace(MeshPtr &mesh);

        inline size_t NumElements()
        {
            return nElements;
        }

        inline double nodeRatio(double d)
        {
            // Compute diagonal distance from corner to corner
            Vector3 diag = maxCoords - minCoords;
            double maxCoord = fmax(diag.x, fmax(diag.y, diag.z));
            return diag.norm() / d;
        }

        // Creates an auxilliary DataTree structure for this BVH.
        template <typename Data, typename Init = DefaultInit<Data>>
        DataTreeContainer<Data> *CreateDataTree()
        {
            DataTree<Data> *droot = CreateDataTreeRecursive<Data, Init>();
            return new DataTreeContainer<Data>(droot, numNodesInBranch);
        }

    private:
        size_t nElements;
        double AxisSplittingPlane(std::vector<MassNormalPoint> &points, int axis);
        void averageDataFromChildren();
        void mergeIndicesFromChildren();

        template <typename Data, typename Init = DefaultInit<Data>>
        DataTree<Data> *CreateDataTreeRecursive()
        {
            DataTree<Data> *droot = new DataTree<Data>(this);
            droot->nodeID = nodeID;
            Init::Init(droot->data, this);
            if (nodeType == BVHNodeType::Interior)
            {
                for (BVHNode6D *child : children)
                {
                    DataTree<Data> *childData = child->CreateDataTreeRecursive<Data, Init>();
                    droot->children.push_back(childData);
                }
            }
            
            return droot;
        }
    };

    BVHNode6D *Create6DBVHFromMeshFaces(MeshPtr &mesh, GeomPtr &geom, double theta);
    BVHNode6D *Create6DBVHFromMeshVerts(MeshPtr &mesh, GeomPtr &geom, double theta);

} // namespace rsurfaces
