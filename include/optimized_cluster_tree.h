#pragma once

#include "bct_kernel_type.h"
#include "optimized_bct_types.h"

namespace rsurfaces
{

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

        ~OptimizedClusterTree()
        {

            // pointer arrays come at the cost of manual deallocation...

            mreal_free(P_in);
            mreal_free(P_out);

            mreal_free(C_in);
            mreal_free(C_out);

            mreal_free(C_squared_radius);

            mint_free(leaf_clusters);
            mint_free(leaf_cluster_lookup);
            mint_free(leaf_cluster_ptr);

            mint_free(inverse_ordering);
            mint_free(P_ext_pos);

            mint_free(C_begin);
            mint_free(C_end);
            mint_free(C_depth);
            mint_free(C_left);
            mint_free(C_right);

            for (mint k = 0; k < static_cast<mint>(P_coords.size()); ++k)
            {
                mreal_free(P_coords[k]);
            }

            for (mint k = 0; k < static_cast<mint>(C_coords.size()); ++k)
            {
                mreal_free(C_coords[k]);
            }

            for (mint k = 0; k < static_cast<mint>(P_near.size()); ++k)
            {
                mreal_free(P_near[k]);
            }

            for (mint k = 0; k < static_cast<mint>(C_far.size()); ++k)
            {
                mreal_free(C_far[k]);
            }

            for (mint k = 0; k < static_cast<mint>(P_min.size()); ++k)
            {
                mreal_free(P_min[k]);
            }

            for (mint k = 0; k < static_cast<mint>(P_max.size()); ++k)
            {
                mreal_free(P_max[k]);
            }

            for (mint k = 0; k < static_cast<mint>(C_min.size()); ++k)
            {
                mreal_free(C_min[k]);
            }

            for (mint k = 0; k < static_cast<mint>(C_max.size()); ++k)
            {
                mreal_free(C_max[k]);
            }

            //            for( mint k = 0; k < moment_count; ++ k )
            //            {
            //                mreal_free(P_moments[k]);
            //            }
            //
            //            for( mint k = 0; k < moment_count; ++ k )
            //            {
            //                mreal_free(C_moments[k]);
            //            }
        };

        inline void UpdateWithNewPositions(MeshPtr &mesh, GeomPtr &geom)
        {
            int nVertices = mesh->nVertices();
            int nFaces = mesh->nFaces();
            FaceIndices fInds = mesh->getFaceIndices();
            VertexIndices vInds = mesh->getVertexIndices();

            double athird = 1. / 3.;

            if( far_dim == 7)
            {
                std::vector<double> P_near_(7 * nFaces);
                std::vector<double> P_far_(7 * nFaces);
                
                for (auto face : mesh->faces())
                {
                    int i = fInds[face];
                    
                    GCHalfedge he = face.halfedge();
                    
                    int i0 = vInds[he.vertex()];
                    int i1 = vInds[he.next().vertex()];
                    int i2 = vInds[he.next().next().vertex()];
                    Vector3 p1 = geom->inputVertexPositions[i0];
                    Vector3 p2 = geom->inputVertexPositions[i1];
                    Vector3 p3 = geom->inputVertexPositions[i2];
                    
                    P_far_[7 * i + 0] = P_near_[7 * i + 0] = geom->faceAreas[face];
                    P_far_[7 * i + 0] = P_near_[7 * i + 1] = athird * (p1.x + p2.x + p3.x);
                    P_far_[7 * i + 0] = P_near_[7 * i + 2] = athird * (p1.y + p2.y + p3.y);
                    P_far_[7 * i + 0] = P_near_[7 * i + 3] = athird * (p1.z + p2.z + p3.z);
                    P_far_[7 * i + 0] = P_near_[7 * i + 4] = geom->faceNormals[face].x;
                    P_far_[7 * i + 0] = P_near_[7 * i + 5] = geom->faceNormals[face].y;
                    P_far_[7 * i + 0] = P_near_[7 * i + 6] = geom->faceNormals[face].z;
                }
                SemiStaticUpdate( &P_near_[0], &P_far_[0] );
            }
            else
            {
                std::vector<double> P_near_(7 * nFaces);
                std::vector<double> P_far_(10 * nFaces);
                
                for (auto face : mesh->faces())
                {
                    int i = fInds[face];
                    
                    GCHalfedge he = face.halfedge();
                    
                    int i0 = vInds[he.vertex()];
                    int i1 = vInds[he.next().vertex()];
                    int i2 = vInds[he.next().next().vertex()];
                    Vector3 p1 = geom->inputVertexPositions[i0];
                    Vector3 p2 = geom->inputVertexPositions[i1];
                    Vector3 p3 = geom->inputVertexPositions[i2];
                    
                    P_far_[10 * i + 0] = P_near_[7 * i + 0] = geom->faceAreas[face];
                    P_far_[10 * i + 0] = P_near_[7 * i + 0] = athird * (p1.x + p2.x + p3.x);
                    P_far_[10 * i + 0] = P_near_[7 * i + 0] = athird * (p1.y + p2.y + p3.y);
                    P_far_[10 * i + 0] = P_near_[7 * i + 0] = athird * (p1.z + p2.z + p3.z);
                    mreal n1 = geom->faceNormals[face].x;
                    mreal n2 = geom->faceNormals[face].y;
                    mreal n3 = geom->faceNormals[face].z;
                    
                    P_near_[7 * i + 4] = n1;
                    P_near_[7 * i + 5] = n2;
                    P_near_[7 * i + 6] = n3;
                    
                    P_far_[10 * i + 4] = n1 * n1;
                    P_far_[10 * i + 5] = n1 * n2;
                    P_far_[10 * i + 6] = n1 * n3;
                    P_far_[10 * i + 7] = n2 * n2;
                    P_far_[10 * i + 8] = n2 * n3;
                    P_far_[10 * i + 9] = n3 * n3;

                }
                SemiStaticUpdate( &P_near_[0], &P_far_[0] );
            }

        }

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

        //    // TODO: Not nearly as fast as I'd like it to be
        void PercolateUp(const mint C, const mint free_thread_count);

        //    // TODO: Not nearly as fast as I'd like it to be
        void PercolateDown(const mint C, const mint free_thread_count);

        void CollectDerivatives( mreal * restrict const P_D_near_output ); // collect only near field data
        
        void CollectDerivatives( mreal * restrict const P_D_near_output, mreal * restrict const P_D_far_output );

    private:
        
        void computeClusterData(const mint C, const mint free_thread_count); // helper function for ComputeClusterData
        
        // Updates only the computational data (primitive/cluster areas, centers of mass and normals).
        // All data related to clustering or multipole acceptance criteria remain are unchanged, as well
        // as the preprocessor and postprocessor matrices (that are needed for matrix-vector multiplies of the BCT.)
        void SemiStaticUpdate( const mreal * restrict const P_near_, const mreal * restrict const P_far_ );

    }; //OptimizedClusterTree
} // namespace rsurfaces
