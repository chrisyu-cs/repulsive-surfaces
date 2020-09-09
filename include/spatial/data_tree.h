#pragma once

#include "bvh_types.h"

namespace rsurfaces
{
    template <typename Data>
    class DefaultInit
    {
    public:
        static void Init(Data &data, BVHNode6D *node) {}
    };

    // An auxilliary tree data structure associated with a BVH,
    // which can store data at each node without it being included
    // in the BVH class itself.
    template <typename Data>
    class DataTree
    {
    public:
        // A DataTree points back to the BVH node it was created from
        DataTree(BVHNode6D *node_)
        {
            node = node_;
        }

        ~DataTree()
        {
            for (DataTree<Data> *child : children)
            {
                delete child;
            }
        }

        // Recursively counts the number of nodes in this tree;
        // generally shouldn't use this unless there's no alternative.
        size_t numNodes()
        {
            size_t total = 1;
            for (DataTree<Data> *child : children)
            {
                total += child->numNodes();
            }
            return total;
        }

        Data data;
        BVHNode6D *node;
        std::vector<DataTree<Data> *> children;
        size_t nodeID;
    };

    // A class that just wraps a DataTree along with an array that indexed
    // node IDs to the actual nodes. This lets you go from a BVH node
    // to its associated data node by going through its ID.
    template <typename Data>
    class DataTreeContainer
    {
    public:
        DataTree<Data> *tree;
        std::vector<DataTree<Data> *> byIndex;

        DataTreeContainer(DataTree<Data> *t, size_t nNodes)
            : byIndex(nNodes)
        {
            tree = t;
        }

        ~DataTreeContainer()
        {
            byIndex.clear();
            delete tree;
        }

        DataTree<Data> *GetDataNode(BVHNode6D *bvhNode);
    };
} // namespace rsurfaces
