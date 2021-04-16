#pragma once

#include "bct_kernel_type.h"
#include "optimized_bct_types.h"

namespace rsurfaces
{

    enum class TreePercolationAlgorithm
    {
        Tasks,
        Sequential,
        Chunks
    };
    
    struct Cluster2 // slim POD container to hold only the data relevant for the construction phase in the tree, before it is serialized
    {
    public:
        Cluster2(){};

        ~Cluster2(){
//            delete left;
//            delete right;
        };

        Cluster2(mint begin_, mint end_, mint depth_);

        mint begin = 0; // position of first triangle in cluster relative to array ordering
        mint end = 0;   // position behind last triangle in cluster relative to array ordering
        mint depth = 0; // depth within the tree -- not absolutely necessary but nice to have for plotting images
        mint max_depth = 0; // used to compute the maximal depth in the tree
        mint descendant_count = 0;
        mint descendant_leaf_count = 0;
        Cluster2 *left = nullptr;
        Cluster2 *right = nullptr;
    }; //Cluster2

    class OptimizedClusterTree // binary cluster tree; layout mostly in Struct of Array fashion in order to prepare SIMDization. Note SIMDized, yet, though.
    {
    public:
        OptimizedClusterTree(){};

        // Solving interface problems by using standard types
        // This way, things are easier to port. For example, I can call this from Mathematica for faster debugging.

        OptimizedClusterTree(
            const mreal * restrict const P_coords_, // coordinates per primitive used for clustering; assumed to be of size primitive_count x dim
            const mint primitive_count_,
            const mint dim_,
            const mreal * restrict const P_hull_coords_, // points that define the convex hulls of primitives; assumed to be array of size primitive_count x hull_count x dim
            const mint hull_count_,
            const mreal * restrict const P_near_, // data used actual interaction computation; assumed to be of size primitive_count x near_dim. For a triangle mesh in 3D, we want to feed each triangles i), area ii) barycenter and iii) normal as a 1 + 3 + 3 = 7 vector
            const mint near_dim_,
            const mreal * restrict const P_far_, // data used actual interaction computation; assumed to be of size primitive_count x far_dim. For a triangle mesh in 3D, we want to feed each triangles i), area ii) barycenter and iii) orthoprojector onto normal space as a 1 + 3 + 6 = 10 vector
            const mint far_dim_,
            //                    const mreal * const restrict P_moments_,          // Interface to deal with higher order multipole expansion. Not used, yet.
            //                    const mint moment_count_,
            const mint max_buffer_dim_,
            const mint * restrict const ordering_, // A suggested preordering of primitives; this gets applied before the clustering begins in the hope that this may improve the sorting within a cluster --- at least in the top level(s). This could, e.g., be the ordering obtained by a tree for  similar data set.
            const mint split_threshold_,          // split a cluster if has this many or more primitives contained in it
            MKLSparseMatrix &DiffOp,              // Asking now for MKLSparseMatrix instead of EigenMatrixCSR as input
            MKLSparseMatrix &AvOp                 // Asking now for MKLSparseMatrix instead of EigenMatrixCSR as input
        );

        mint split_threshold = 8; // leaf clusters will contain split_threshold triangles or less; split_threshold = 8 might be good for cache.
        mint dim = 3;
        mint near_dim = 7; // = 1 + 3 + 3 for weight, center, normal, stored consecutively
        mint far_dim = 10; // = 1 + 3 + 3 * (3 + 1)/2 for weight, center, projector, stored consecutively
        mint hull_count = 3;
        mint tree_thread_count = 1;
        mint thread_count = 1;

        mint primitive_count = 0;
        mint cluster_count = 0;
        mint leaf_cluster_count = 0;
        mint max_buffer_dim = 0;
        mint buffer_dim = 0;
        //        mint moment_count = 22;

        mint *restrict P_ext_pos = nullptr;        // Reordering of primitives; crucial for communication with outside world
        mint *restrict inverse_ordering = nullptr; // Inverse ordering of the above; crucial for communication with outside world
                                         //    A_Vector<mint> P_leaf;               // Index of the leaf cluster to which the primitive belongs

        // "C_" stands for "cluster", "P_" stands for "primitive"

        mint *restrict C_begin = nullptr;
        mint *restrict C_end = nullptr;
        mint *restrict C_depth = nullptr;
        mint *restrict C_next = nullptr;
        mint *restrict C_left = nullptr;  // list of index of left children;  entry is -1 if no child is present
        mint *restrict C_right = nullptr; // list of index of right children; entry is -1 if no child is present

        // Primitive double data, stored in Structure of Arrays fashion

        A_Vector<mreal *> P_near;   //weight, center, normal, stored consecutively; assumed to be matrix of size near_dim x primitive_count!
        A_Vector<mreal *> P_far;   //weight, center, projector, stored consecutively; assumed to be matrix of size far_dim x primitive_count!
        A_Vector<mreal *> P_coords; //clustering coordinates, stored as dim x primitive_count matrix
        A_Vector<mreal *> P_min;    //lower bounding box point, stored as dim x primitive_count matrix
        A_Vector<mreal *> P_max;    //upper bounding box point, stored as dim x n matrix
                                    //        A_Vector<mreal * restrict> P_moments;
        mreal *restrict P_in = nullptr;
        mreal *restrict P_out = nullptr;
        //        mreal * restrict P_moment_buffer = nullptr;

        // Cluster double data, stored in Structure of Arrays fashion

        A_Vector<mreal *> C_far;   //weight, center, normal, stored consecutively; assumed to be matrix of size data_dim x n
        A_Vector<mreal *> C_coords; //clustering coordinate
        A_Vector<mreal *> C_min;
        A_Vector<mreal *> C_max;
        //        A_Vector<mreal * restrict> C_moments;
        mreal *restrict C_in = nullptr;
        mreal *restrict C_out = nullptr;
        //        mreal * restrict C_moment_buffer = nullptr;

        mreal *restrict C_squared_radius = nullptr;

        mint *restrict leaf_clusters = nullptr;
        mint *restrict leaf_cluster_lookup = nullptr;
        mint *restrict leaf_cluster_ptr = nullptr; // point to __end__ of each leaf cluster

        A_Vector<A_Vector<mreal>> P_D_near;
        A_Vector<A_Vector<mreal>> P_D_far;
        A_Vector<A_Vector<mreal>> C_D_far;

        //        mint scratch_size = 12;
        //        A_Vector<A_Vector<mreal>> scratch;

        MKLSparseMatrix hi_pre;
        MKLSparseMatrix hi_post;

        MKLSparseMatrix lo_pre;
        MKLSparseMatrix lo_post;

        MKLSparseMatrix P_to_C;
        MKLSparseMatrix C_to_P;

        TreePercolationAlgorithm tree_perc_alg = TreePercolationAlgorithm::Tasks;
        A_Vector<A_Vector<mint>> chunk_roots;
        mint tree_max_depth = 0;
        bool chunks_prepared = false;
        
        ~OptimizedClusterTree()
        {
            ptic("~OptimizedClusterTree");
            // pointer arrays come at the cost of manual deallocation...
            
            #pragma omp parallel
            {
                #pragma omp single
                {
//                    #pragma omp task
//                    {
//                        for( mint k = 0; k < moment_count; ++ k )
//                        {
//                            safe_free(P_moments[k]);
//                        }
//                    }
//
//                    #pragma omp task
//                    {
//                        for( mint k = 0; k < moment_count; ++ k )
//                        {
//                            safe_free(C_moments[k]);
//                        }
//                    }
                
                    #pragma omp task
                    {
                        for (mint k = 0; k < static_cast<mint>(P_coords.size()); ++k)
                        {
                            safe_free(P_coords[k]);
                        }
                    }

                    #pragma omp task
                    {
                        for (mint k = 0; k < static_cast<mint>(C_coords.size()); ++k)
                        {
                            safe_free(C_coords[k]);
                        }
                    }

                    #pragma omp task
                    {
                        for (mint k = 0; k < static_cast<mint>(P_near.size()); ++k)
                        {
                            safe_free(P_near[k]);
                        }
                    }

                    #pragma omp task
                    {
                        for (mint k = 0; k < static_cast<mint>(C_far.size()); ++k)
                        {
                            safe_free(C_far[k]);
                        }
                    }

                    #pragma omp task
                    {
                        for (mint k = 0; k < static_cast<mint>(P_min.size()); ++k)
                        {
                            safe_free(P_min[k]);
                        }
                    }

                    #pragma omp task
                    {
                        for (mint k = 0; k < static_cast<mint>(P_max.size()); ++k)
                        {
                            safe_free(P_max[k]);
                        }
                    }

                    #pragma omp task
                    {
                        for (mint k = 0; k < static_cast<mint>(C_min.size()); ++k)
                        {
                            safe_free(C_min[k]);
                        }
                    }

                    #pragma omp task
                    {
                        for (mint k = 0; k < static_cast<mint>(C_max.size()); ++k)
                        {
                            safe_free(C_max[k]);
                        }
                    }

                    #pragma omp task
                    {
                        safe_free(P_in);
                    }

                    #pragma omp task
                    {
                        safe_free(P_out);
                    }

                    #pragma omp task
                    {
                        safe_free(C_in);
                    }

                    #pragma omp task
                    {
                        safe_free(C_out);
                    }

                    #pragma omp task
                    {
                        safe_free(C_squared_radius);
                    }

                    #pragma omp task
                    {
                        safe_free(leaf_clusters);
                    }

                    #pragma omp task
                    {
                        safe_free(leaf_cluster_lookup);
                    }
                    #pragma omp task

                    {
                        safe_free(leaf_cluster_ptr);
                    }

                    #pragma omp task
                    {
                        safe_free(inverse_ordering);
                    }

                    #pragma omp task
                    {
                        safe_free(P_ext_pos);
                    }

                    #pragma omp task
                    {
                        safe_free(C_begin);
                    }

                    #pragma omp task
                    {
                        safe_free(C_end);
                    }

                    #pragma omp task
                    {
                        safe_free(C_depth);
                    }

                    #pragma omp task
                    {
                        safe_free(C_next);
                    }

                    #pragma omp task
                    {
                        safe_free(C_left);
                    }

                    #pragma omp task
                    {
                        safe_free(C_right);
                    }

                    }
            }
            ptoc("~OptimizedClusterTree");
        };

        void SplitCluster(Cluster2 * const C, const mint free_thread_count);

        void Serialize(Cluster2 * const C, const mint ID, const mint leaf_before_count, const mint free_thread_count);

        void ComputePrimitiveData(
            const mreal * restrict const P_hull_coords_,
            const mreal * restrict const P_near_,
            const mreal * restrict const P_far_
            //                                  , const mreal * const  restrict P_moments_
        ); // copy, reordering and computing bounding boxes

        void ComputeClusterData();

        void RequireBuffers(const mint cols);

        void ComputePrePost(MKLSparseMatrix &DiffOp, MKLSparseMatrix &AvOp);

        void CleanseBuffers();

        void CleanseD();

        void Pre(Eigen::MatrixXd &input, BCTKernelType type);

        void Pre(mreal *input, const mint cols, BCTKernelType type);

        void Post(Eigen::MatrixXd &output, BCTKernelType type, bool addToResult = false);

        void Post(mreal *output, const mint cols, BCTKernelType type, bool addToResult = false);

        void PercolateUp();
        
        void PercolateDown();
        
        void PrepareChunks();
        
        // some prototype
        void PercolateUp_Chunks();

        // some prototype
        void PercolateDown_Chunks();
        
        // TODO: Not nearly as fast as I'd like it to be; not scalable!
        // recusive algorithm parallelized by OpenMP tasks
        void PercolateUp_Tasks(const mint C, const mint free_thread_count);

        // TODO: Not nearly as fast as I'd like it to be; not scalable!
        // recusive algorithm parallelized by OpenMP tasks
        void PercolateDown_Tasks(const mint C, const mint free_thread_count);
        
        // TODO: use a stack for recursion instead of the program stack?
        // sequential, recursive algorithm
        void PercolateUp_Seq(const mint C);

        // TODO: use a stack for recursion instead of the program stack?
        // sequential, recursive algorithm
        void PercolateDown_Seq(const mint C);

        void CollectDerivatives( mreal * restrict const P_D_near_output ); // collect only near field data
        
        void CollectDerivatives( mreal * restrict const P_D_near_output, mreal * restrict const P_D_far_output );

        // Updates only the computational data (primitive/cluster areas, centers of mass and normals).
        // All data related to clustering or multipole acceptance criteria remain are unchanged, as well
        // as the preprocessor and postprocessor matrices (that are needed for matrix-vector multiplies of the BCT.)
        void SemiStaticUpdate( const mreal * restrict const P_near_, const mreal * restrict const P_far_ );
        
        void PrintToFile(std::string filename = "./OptimizedClusterTree.tsv");
        
    private:
        
        void computeClusterData(const mint C, const mint free_thread_count); // helper function for ComputeClusterData

        bool prepareChunks( mint C, mint last, mint thread);

        
    }; //OptimizedClusterTree
} // namespace rsurfaces
