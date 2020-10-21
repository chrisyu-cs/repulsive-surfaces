#pragma once

#include "bvh_types.h"
#include "bvh_data.h"
#include "data_tree.h"
#include "helpers.h"

#define BVH_N_CHILDREN 2

namespace rsurfaces
{
    // Find the minimum coordinate of the bounding box of this face
    inline Vector3 minCoordOfFace(const GCFace &f, GeomPtr const &geom)
    {
        Vector3 minCoord = geom->inputVertexPositions[f.halfedge().vertex()];
        for (GCVertex v : f.adjacentVertices())
        {
            minCoord = vectorMin(minCoord, geom->inputVertexPositions[v]);
        }
        return minCoord;
    }

    // Find the maximum coordinate of the bounding box of this face
    inline Vector3 maxCoordOfFace(const GCFace &f, GeomPtr const &geom)
    {
        Vector3 maxCoord = geom->inputVertexPositions[f.halfedge().vertex()];
        for (GCVertex v : f.adjacentVertices())
        {
            maxCoord = vectorMax(maxCoord, geom->inputVertexPositions[v]);
        }
        return maxCoord;
    }

    class BVHNode6D
    {
    public:
        // Build a BVH of the given points
        BVHNode6D(std::vector<MassNormalPoint> &points, int axis);
        ~BVHNode6D();

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
        // Children
        BVHNode6D *children[BVH_N_CHILDREN];

        // Store the list of all indices of elements in this cluster
        std::vector<size_t> clusterIndices;

        inline GCFace getSingleFace(MeshPtr &mesh)
        {
            return mesh->face(elementID);
        }

        // Assign unique IDs to all nodes in this tree
        size_t assignIDsRecursively(size_t startID);
        // Recursively recompute all centers of mass in this tree
        void recomputeCentersOfMass(MeshPtr const &mesh, GeomPtr const &geom);

        inline bool isAdmissibleFrom(Vector3 atPos, double thresholdTheta)
        {
            if (nodeType == BVHNodeType::Leaf)
            {
                if (centerOfMass == atPos)
                    return false;
                else
                    return true;
            }
            else if (nodeType == BVHNodeType::Interior)
            {
                if (boxContainsPoint(atPos))
                {
                    return false;
                }
                double d = norm(centerOfMass - atPos);
                return nodeRatioBox(minCoords, maxCoords, d) < thresholdTheta;
            }
            else
                return true;
        }

        inline bool boxContainsPoint(Vector3 pos)
        {
            bool xOK = (minCoords.x < pos.x) && (pos.x < maxCoords.x);
            bool yOK = (minCoords.y < pos.y) && (pos.y < maxCoords.y);
            bool zOK = (minCoords.z < pos.z) && (pos.z < maxCoords.z);
            return xOK && yOK && zOK;
        }

        void printSummary();
        MassNormalPoint GetMassNormalPoint();

        inline BVHData GetNodeDataAsStruct()
        {
            BVHData data{totalMass, centerOfMass, averageNormal, minCoords, maxCoords, elementID, nodeID, nodeType, numNodesInBranch, nElements, {0, 0}};
            if (nodeType == BVHNodeType::Interior)
            {
                data.child[0] = children[0]->nodeID;
                data.child[1] = children[1]->nodeID;
            }
            return data;
        }

        inline double nodeRatio(double d)
        {
            return nodeRatioBox(minCoords, maxCoords, d);
        }

        template <typename Data>
        void indexNodesForDataTree(DataTreeContainer<Data> *cont, DataTree<Data> *droot)
        {
            // Put the root in the correct spot
            cont->byIndex[droot->nodeID] = droot;
            // Recursively index children
            for (DataTree<Data> *child : droot->children)
            {
                indexNodesForDataTree(cont, child);
            }
        }

        // Creates an auxilliary DataTree structure for this BVH.
        template <typename Data, typename Init = DefaultInit<Data>>
        DataTreeContainer<Data> *CreateDataTree()
        {
            DataTree<Data> *droot = CreateDataTreeRecursive<Data, Init>();
            DataTreeContainer<Data> *cont = new DataTreeContainer<Data>(droot, numNodesInBranch);
            indexNodesForDataTree(cont, droot);
            return cont;
        }

    private:
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

    inline MassNormalPoint meshFaceToBody(const GCFace &f, GeomPtr const &geom, FaceIndices &indices)
    {
        Vector3 pos = faceBarycenter(geom, f);
        double mass = geom->faceArea(f);
        Vector3 n = geom->faceNormal(f);

        Vector3 minCoord = minCoordOfFace(f, geom);
        Vector3 maxCoord = maxCoordOfFace(f, geom);

        return MassNormalPoint{mass, n, pos, minCoord, maxCoord, indices[f]};
    }

    template <typename MPtr, typename GPtr>
    BVHNode6D *Create6DBVHFromMeshFaces(MPtr const &mesh, GPtr const &geom)
    {
        std::vector<MassNormalPoint> verts(mesh->nFaces());
        FaceIndices indices = mesh->getFaceIndices();

        // Loop over all the vertices
        for (const GCFace &f : mesh->faces())
        {
            if (f.isBoundaryLoop())
            {
                continue;
            }
            MassNormalPoint curBody = meshFaceToBody(f, geom, indices);
            // Put vertex body into full list
            verts[curBody.elementID] = curBody;
        }

        BVHNode6D *tree = new BVHNode6D(verts, 0);
        tree->assignIDsRecursively(0);
        return tree;
    }

    inline MassNormalPoint meshVertToBody(const GCVertex &v, GeomUPtr const &geom, VertexIndices &indices)
    {
        Vector3 pos = geom->inputVertexPositions[v];
        double mass = geom->vertexDualAreas[v];
        Vector3 n = geom->vertexNormals[v];

        Vector3 minCoord = pos;
        Vector3 maxCoord = pos;

        return MassNormalPoint{mass, n, pos, minCoord, maxCoord, indices[v]};
    }

    template <typename MPtr, typename GPtr>
    BVHNode6D *Create6DBVHFromMeshVerts(MPtr const &mesh, GPtr const &geom)
    {
        std::vector<MassNormalPoint> verts(mesh->nVertices());
        VertexIndices indices = mesh->getVertexIndices();

        // Loop over all the vertices
        for (const GCVertex &v : mesh->vertices())
        {
            MassNormalPoint curBody = meshVertToBody(v, geom, indices);
            // Put vertex body into full list
            verts[curBody.elementID] = curBody;
        }

        BVHNode6D *tree = new BVHNode6D(verts, 0);
        tree->assignIDsRecursively(0);
        return tree;
    }

    template <typename Data>
    DataTree<Data> *DataTreeContainer<Data>::GetDataNode(BVHNode6D *bvhNode)
    {
        return byIndex[bvhNode->nodeID];
    }

} // namespace rsurfaces
