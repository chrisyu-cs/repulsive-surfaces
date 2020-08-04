#pragma once

#include "bvh_types.h"

namespace rsurfaces {
    template<typename Data>
    class DefaultInit {
        public:
        static void Init(Data &data, BVHNode6D* node) {}
    };
    
    template<typename Data>
    class DataTree {
        public:
        DataTree(BVHNode6D* node_) {
            node = node_;
        }
        Data data;
        BVHNode6D* node;
        std::vector<DataTree<Data>*> children;
    };
}


