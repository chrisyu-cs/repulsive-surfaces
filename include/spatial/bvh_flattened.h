#pragma once

#include "spatial/bvh_6d.h"

namespace rsurfaces
{
    class BVHFlattened
    {
    public:
        BVHFlattened(BVHNode6D *root);
        std::vector<BVHData> nodes;
        
    private:
        void FlattenNodes(BVHNode6D *root);
    };
} // namespace rsurfaces
