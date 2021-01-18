#pragma once

#include "bct_kernel_type.h"
#include "block_cluster_tree2_types.h"

namespace rsurfaces
{

    struct Cluster2 // slim POD container to hold only the data relevant for the construction phase in the tree, before it is serialized
    {
    public:
        Cluster2(){};

        ~Cluster2(){};

        Cluster2(mint begin_, mint end_, mint depth_);

        mint begin = 0; // position of first triangle in cluster relative to array ordering
        mint end = 0;   // position behind last triangle in cluster relative to array ordering
        mint depth = 0; // depth within the tree -- not absolutely necessary but nice to have for plotting images
        mint descendant_count = 0;
        mint descendant_leaf_count = 0;
        std::shared_ptr<Cluster2> left = nullptr;
        std::shared_ptr<Cluster2> right = nullptr;
    }; //Cluster2
    
    class ClusterTree2 // binary cluster tree; layout mostly in Struct of Array fashion in order to prepare SIMDization. Note SIMDized, yet, though.
    {
    public:
        ClusterTree2(){};

        ~ClusterTree2(){};

        // Solving interface problems by using standard types
        // This way, things are easier to port. For example, I can call this from Mathematica for faster debugging.

        ClusterTree2(
            const mreal *const restrict P_coords_, // coordinates per primitive used for clustering; assumed to be of size primitive_count x dim
            const mint primitive_count_,
            const mint dim_,
            const mreal *const restrict P_hull_coords_, // points that define the convex hulls of primitives; assumed to be array of size primitive_count x hull_count x dim
            const mint hull_count_,
            const mreal *const restrict P_data_, // data used actual interaction computation; assumed to be of size primitive_count x data_dim. For a triangle mesh in 3D, we want to feed each triangles i), area ii) barycenter and iii) normal as a 1 + 3 + 3 = 7 vector
            const mint data_dim_,
            const mint max_buffer_dim_,
            const mint *const restrict ordering_, // A suggested preordering of primitives; this gets applied before the clustering begins in the hope that this may improve the sorting within a cluster --- at least in the top level(s). This could, e.g., be the ordering obtained by a tree for  similar data set.
            const mint split_threshold_,          // split a cluster if has this many or more primitives contained in it
                                                  //                 EigenMatrixCSC & DiffOp,        // CSC
                                                  //                 EigenMatrixCSC & AvOp           // CSC
            EigenMatrixCSR &DiffOp,               // CSR
            EigenMatrixCSR &AvOp                  // CSR
        );

        mint split_threshold = 8; // leaf clusters will contain split_threshold triangles or less; split_threshold = 8 might be good for cache.
        mint dim = 3;
        mint data_dim = 7; // = 1 + 3 + 3 for weight, center, normal, stored consecutively
        mint hull_count = 3;
        mint thread_count = 1;

        mint primitive_count = 0;
        mint cluster_count = 0;
        mint leaf_cluster_count = 0;
        mint max_buffer_dim = 0;
        mint buffer_dim = 0;

        A_Vector<mint> P_ext_pos;        // Reordering of primitives; crucial for communication with outside world
        A_Vector<mint> inverse_ordering; // Inverse ordering of the above; crucial for communication with outside world
        A_Vector<mint> P_leaf;           // Index of the leaf cluster to which the primitive belongs

        // "C_" stands for "cluster", "P_" stands for "primitive"

        A_Vector<mint> C_begin; // list of starting indices for clusters (relative to ordering!)
        A_Vector<mint> C_end;   // list of indices pointing to positions behind last primitives in  clusters (relative to ordering!)
        A_Vector<mint> C_depth;
        A_Vector<mint> C_left;  // list of index of left children;  entry is -1 if no child is present
        A_Vector<mint> C_right; // list of index of right children; entry is -1 if no child is present
                                //    A_Vector<mint>  C_ID;           // seems pointless, but is used for sparse arrays

        // Primitive double data, stored in Structure of Arrays fashion

        A_Vector<A_Vector<mreal>> P_data;   //weight, center, normal, stored consecutively; assumed to be matrix of size data_dim x n!
        A_Vector<A_Vector<mreal>> P_coords; //clustering coordinates, stored as dim x n matrix
        A_Vector<A_Vector<mreal>> P_min;    //lower bounding box point, stored as dim x n matrix
        A_Vector<A_Vector<mreal>> P_max;    //upper bounding box point, stored as dim x n matrix
        A_Vector<mreal> P_in;
        A_Vector<mreal> P_out;

        // Cluster double data, stored in Structure of Arrays fashion

        A_Vector<A_Vector<mreal>> C_data;   //weight, center, normal, stored consecutively; assumed to be matrix of size data_dim x n
        A_Vector<A_Vector<mreal>> C_coords; //clustering coordinate
        A_Vector<A_Vector<mreal>> C_min;
        A_Vector<A_Vector<mreal>> C_max;
        A_Vector<mreal> C_squared_radius;
        A_Vector<mreal> C_in;
        A_Vector<mreal> C_out;
        //    A_Vector<mreal> C_one;          // constant vector of ones.

        A_Vector<mint> leaf_clusters;
        A_Vector<mint> leaf_cluster_lookup;
        A_Vector<mint> leaf_cluster_ptr; // point to __end__ of each leaf cluster

        MKLSparseMatrix hi_pre;
        MKLSparseMatrix hi_post;

        MKLSparseMatrix lo_pre;
        MKLSparseMatrix lo_post;

        MKLSparseMatrix P_to_C;
        MKLSparseMatrix C_to_P;

        void SplitCluster(Cluster2 *const C, const mint free_thread_count);

        void Serialize(Cluster2 *const C, const mint ID, const mint leaf_before_count, const mint free_thread_count);

        void ComputePrimitiveData(const mreal *const restrict P_hull_coords_, const mreal *const restrict P_data_); // copy, reordering and computing bounding boxes

        void ComputeClusterData();

        void PrepareBuffers(const mint cols);

        void CleanseBuffers();

        void Pre(Eigen::MatrixXd &input, BCTKernelType type);

        void Pre(mreal *input, const mint cols, BCTKernelType type);

        void Post(Eigen::MatrixXd &output, BCTKernelType type, bool addToResult = false);

        void Post(mreal *output, const mint cols, BCTKernelType type, bool addToResult = false);

        //    // TODO: Not nearly as fast as I'd like it to be
        void PercolateUp(const mint C, const mint free_thread_count);

        //    // TODO: Not nearly as fast as I'd like it to be
        void PercolateDown(const mint C, const mint free_thread_count);

        //private:

        void computeClusterData(const mint C, const mint free_thread_count); // helper function for ComputeClusterData
    };                                                                       //ClusterTree2
} // namespace rsurfaces
