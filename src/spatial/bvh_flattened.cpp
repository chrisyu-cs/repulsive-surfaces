#include "spatial/bvh_flattened.h"

namespace rsurfaces
{
    BVHFlattened::BVHFlattened(BVHNode6D *root)
    {
        nodes.resize(root->numNodesInBranch);
        FlattenNodes(root);
    }

    void BVHFlattened::FlattenNodes(BVHNode6D *root)
    {
        if (root->nodeType != BVHNodeType::Empty)
        {
            nodes[root->nodeID] = root->GetNodeDataAsStruct();
        }

        if (root->nodeType == BVHNodeType::Interior)
        {
            for (BVHNode6D *child : root->children)
            {
                FlattenNodes(child);
            }
        }
    }
} // namespace rsurfaces
