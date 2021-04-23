#include "optimized_cluster_tree.h"

namespace rsurfaces
{
    mint OptimizedClusterTreeOptions::split_threshold = 8;
    bool OptimizedClusterTreeOptions::use_old_prepost = false;
    TreePercolationAlgorithm OptimizedClusterTreeOptions::tree_perc_alg = TreePercolationAlgorithm::Chunks;
    
    Cluster2::Cluster2(mint begin_, mint end_, mint depth_)
    {
        begin = begin_;                 // first primitive in cluster
        end = end_;                     // first primitive after cluster
        depth = depth_;                 // depth of cluster in BVH
        max_depth = depth_;             // maximal depth of all descendants of this cluster
        descendant_count = 0;           // number of descendents of cluster, _this cluster included_
        descendant_leaf_count = 0;      // number of leaf descendents of cluster
        left = nullptr;                 // left child
        right = nullptr;                // right child
    }; // struct Cluster2

    // Solving interface problems by using standard types. This way, things are easier to port. For example, I can call this from Mathematica for faster debugging.
    OptimizedClusterTree::OptimizedClusterTree(
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
       const mint * restrict const ordering_, // A suggested preordering of primitives; this gets applied before the clustering begins in the hope that this may improve the sorting within a cluster --- at least in the top level(s). This could, e.g., be the ordering obtained by a tree for  similar data set.
       MKLSparseMatrix &DiffOp,
       MKLSparseMatrix &AvOp
    )
    {
        ptic("OptimizedClusterTree::OptimizedClusterTree");

        use_old_prepost = OptimizedClusterTreeOptions::use_old_prepost;
        
        primitive_count = primitive_count_;
        hull_count = hull_count_;
        dim = dim_;
        near_dim = near_dim_;
        far_dim = far_dim_;
//        moment_count = moment_count_;

//        scratch_size = 12;
        mint nthreads;
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }

        tree_thread_count = std::max( static_cast<mint>(1), nthreads );
             thread_count = std::max( static_cast<mint>(1), nthreads );
        
        split_threshold = std::max( static_cast<mint>(1), OptimizedClusterTreeOptions::split_threshold );

        P_coords = A_Vector<mreal * >( dim, nullptr );

        #pragma omp parallel for
        for( mint k = 0; k < dim; ++k)
        {
            safe_alloc( P_coords[k], primitive_count );
        }

        safe_alloc( P_ext_pos, primitive_count );

        #pragma omp parallel for num_threads(thread_count)  shared( P_coords, P_ext_pos, P_coords_, dim, primitive_count, ordering_ )
        for( mint i=0; i < primitive_count; ++i )
        {
            mint j = ordering_[i];
            P_ext_pos[i] = j;
            for( mint k = 0, last = dim; k < last; ++k )
            {
                P_coords[k][i] = P_coords_[ dim * j + k ];
            }
        }

        ptic("SplitCluster");

        Cluster2 * root = new Cluster2 ( 0, primitive_count, 0 );

        #pragma omp parallel num_threads(tree_thread_count)  shared( root, P_coords, P_ext_pos, tree_thread_count)
        {
            #pragma omp single nowait
            {
                SplitCluster( root, tree_thread_count );
            }
        }
        ptoc("SplitCluster");

        ptic("Bunch of allocations");

        cluster_count = root->descendant_count;
        leaf_cluster_count = root->descendant_leaf_count;
        tree_max_depth = root->max_depth;
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                #pragma omp task
                {
                    mint s = std::max( dim * dim, far_dim);
                    RequireBuffers( std::max( s, max_buffer_dim ) );
                }
                #pragma omp task
                {
                    safe_alloc( C_left, cluster_count );
                }
                #pragma omp task
                {
                    safe_alloc( C_right, cluster_count );
                }
                #pragma omp task
                {
                    safe_alloc( C_begin, cluster_count );
                }
                #pragma omp task
                {
                    safe_alloc( C_end, cluster_count );
                }
                #pragma omp task
                {
                    safe_alloc( C_depth, cluster_count );
                }
                #pragma omp task
                {
                    safe_alloc( C_next, cluster_count );
                }
                #pragma omp task
                {
                    safe_alloc( leaf_clusters, leaf_cluster_count );
                }
                #pragma omp task
                {
                    safe_alloc( leaf_cluster_lookup, cluster_count );
                }
                #pragma omp task
                {
                    safe_alloc( inverse_ordering, primitive_count );
                }
                #pragma omp taskwait
            }
        }
        ptoc("Bunch of allocations");

        ptic("Serialize");
        
        #pragma omp parallel num_threads(tree_thread_count)
        {
            #pragma omp single nowait
            {
                Serialize( root, 0, 0, tree_thread_count );
            }
        }

        delete root;
        
        #pragma omp parallel for
        for( mint i = 0; i < primitive_count; ++i )
        {
            inverse_ordering[P_ext_pos[i]] = i;
        }
        ptoc("Serialize");

        ComputePrimitiveData( P_hull_coords_, P_near_, P_far_ );
//        ComputePrimitiveData( P_hull_coords_, P_near_, P_far_, P_moments_ );

        ComputeClusterData();

        ComputePrePost( DiffOp, AvOp );

        
        ptoc("OptimizedClusterTree::OptimizedClusterTree");
    }; //Constructor


    void OptimizedClusterTree::SplitCluster( Cluster2 * const C, const mint free_thread_count )
    {
        
        mint begin = C->begin;
        mint end = C->end;
        mint cpcount = end - begin;
        
        mint splitdir = -1;
        mreal L, Lmax;
        Lmax = -1.;
        mreal dirmin, dirmax, min, max;
        std::pair< mreal*, mreal* > range;

        // compute splitdir, the longest direction of bounding box
        for( mint k = 0; k < dim; ++k )
        {
            range = std::minmax_element( &P_coords[k][begin], &P_coords[k][end] );
            min = *range.first;
            max = *range.second;
            L =  max - min;
            if( L > Lmax )
            {
                Lmax = L;
                splitdir = k;
                dirmin = min;
                dirmax = max;
            }
        }

        if( (cpcount > split_threshold) && (Lmax > 0.) )
        {
            mreal mid = 0.5 * (dirmin + dirmax);
        
            // swapping points left from mid to the left and determining splitindex
            mint splitindex = begin;
            for( mint i = begin; i < end; ++i )
            {
                if( P_coords[splitdir][i] <= mid )
                {
                    std::swap( P_ext_pos[i], P_ext_pos[splitindex] );
    //                #pragma clang loop vectorize (enable)
                    for( mint k = 0, last = dim; k < last; ++k )
                    {
                        std::swap( P_coords[k][i], P_coords[k][splitindex] );
                    }
                    ++splitindex;
                }
            }

            // create new nodes...
            C->left  = new Cluster2 ( begin, splitindex, C->depth+1 );
            C->right = new Cluster2 ( splitindex, end, C->depth+1 );
            // ... and split them in parallel
            #pragma omp task final(free_thread_count<1)  shared(C, free_thread_count)
            {
                SplitCluster( C->left, free_thread_count/2 );
            }
            #pragma omp task final(free_thread_count<1)  shared(C, free_thread_count)
            {
                SplitCluster( C->right, free_thread_count - free_thread_count/2 );
            }
            #pragma omp taskwait
            
            // collecting statistics for the later serialization
            // counting ourselves as descendant, too!
            C->descendant_count = 1 + C->left->descendant_count + C->right->descendant_count;
            C->descendant_leaf_count = C->left->descendant_leaf_count + C->right->descendant_leaf_count;
            C->max_depth = std::max( C->left->max_depth, C->right->max_depth );
        }
        else
        {
            // count cluster as leaf cluster
            // counting ourselves as descendant, too!
            C->descendant_count = 1;
            C->descendant_leaf_count = 1;
        }
    }; //SplitCluster


    void OptimizedClusterTree::Serialize( Cluster2 * C, mint ID, mint leaf_before_count, mint free_thread_count )
    {
        // enumeration in depth-first order
        C_begin[ID] = C->begin;
        C_end[ID] = C->end;
        C_depth[ID] = C->depth;
        C_next[ID] = ID + C->descendant_count;
        
        if( ( C->left != nullptr ) && ( C->right  != nullptr ) )
        {
            C_left [ID] = ID + 1;
            C_right[ID] = ID + 1 + C->left->descendant_count;
    //
            #pragma omp task final(free_thread_count<1)  shared( C, ID, C_left, C_right ) firstprivate( free_thread_count, leaf_before_count )
            {
                Serialize( C->left, C_left[ID], leaf_before_count, free_thread_count/2 );
            }
            #pragma omp task final(free_thread_count<1)  shared( C, ID, C_left, C_right ) firstprivate( free_thread_count, leaf_before_count )
            {
                Serialize( C->right, C_right[ID], leaf_before_count + C->left->descendant_leaf_count, free_thread_count - free_thread_count/2 );
            }
            #pragma omp taskwait
            
            delete C->left;
            delete C->right;
        }
        else
        {
            C_left [ID] = -1;
            C_right[ID] = -1;
            
            leaf_clusters[leaf_before_count] = ID;
            leaf_cluster_lookup[ID] = leaf_before_count;
        }
    }; //Serialize


    void OptimizedClusterTree::ComputePrimitiveData(
                                           const mreal * restrict const P_hull_coords_,
                                           const mreal * restrict const  P_near_,
                                           const mreal * restrict const  P_far_
//                                           , const mreal * const  restrict P_moments_
                                           ) // reordering and computing bounding boxes
    {
        ptic("OptimizedClusterTree::ComputePrimitiveData");
        
        P_near = A_Vector<mreal * > ( near_dim, nullptr );
        for( mint k = 0; k < near_dim; ++ k )
        {
            safe_alloc( P_near[k], primitive_count );
        }
        
        P_far  = A_Vector<mreal * > ( far_dim , nullptr );
        for( mint k = 0; k < far_dim; ++ k )
        {
            safe_alloc( P_far[k], primitive_count );
        }
        
        P_min = A_Vector<mreal * > ( dim, nullptr );
        P_max = A_Vector<mreal * > ( dim, nullptr );
        for( mint k = 0; k < dim; ++ k )
        {
            safe_alloc( P_min[k], primitive_count );
            safe_alloc( P_max[k], primitive_count );
        }
        
        P_D_near = A_Vector<A_Vector<mreal>> ( thread_count );
        P_D_far  = A_Vector<A_Vector<mreal>> ( thread_count );
        
//        P_moments = A_Vector<mreal * restrict> ( moment_count, nullptr );
//        for( mint k = 0; k < moment_count; ++ k )
//        {
//            safe_alloc( P_moments[k], primitive_count );
//        }
            
        mint hull_size = hull_count * dim;
        
        #pragma omp parallel for shared( thread_count ) schedule( static, 1 )
        for( mint thread = 0; thread < thread_count; ++thread )
        {
            P_D_near[thread] = A_Vector<mreal> ( primitive_count * near_dim );
            P_D_far[thread]  = A_Vector<mreal> ( primitive_count * far_dim );
        }
        
        #pragma omp parallel for shared( P_near, P_far, P_ext_pos, P_min, P_min, P_near_, P_far_, P_hull_coords_, near_dim, far_dim, hull_size, dim )
        for( mint i = 0; i < primitive_count; ++i )
        {
            mreal min, max;
            mint j = P_ext_pos[i];
            for( mint k = 0; k < near_dim; ++k )
            {
                P_near[k][i] = P_near_[ near_dim * j + k];
            }
            
            for( mint k = 0; k < far_dim; ++k )
            {
                P_far [k][i] = P_far_ [ far_dim  * j + k];
            }
            
//            for( mint k = 0; k < moment_count; ++k )
//            {
//                P_moments[k][i] = P_moments_[ moment_count * j + k];
//            }

            // computing bounding boxes of primitives; admittedly, it looks awful
            
            for( mint k = 0; k < dim; ++k )
            {
                min = max = P_hull_coords_[ hull_size * j + dim * 0 + k];
                for( mint h = 1; h < hull_count; ++h )
                {
                    mreal x = P_hull_coords_[ hull_size * j + dim * h + k];
                    min = mymin( min , x );
                    max = mymax( max , x );
                }
                P_min[k][i] = min;
                P_max[k][i] = max;
            }
        }
        
        ptoc("OptimizedClusterTree::ComputePrimitiveData");
    } //ComputePrimitiveData

    void OptimizedClusterTree::ComputeClusterData()
    {
        
        ptic("OptimizedClusterTree::ComputeClusterData");
        
//        scratch = A_Vector<A_Vector<mreal>> ( thread_count );
//        for( mint thread = 0; thread < thread_count; ++thread )
//        {
//            scratch[thread] = A_Vector<mreal> ( scratch_size );
//        }
        
        C_far = A_Vector<mreal * > ( far_dim, nullptr );
        for( mint k = 0; k < far_dim; ++ k )
        {
            safe_alloc( C_far[k], cluster_count, 0. );
        }
        
        C_coords = A_Vector<mreal * > ( dim, nullptr );
        C_min = A_Vector<mreal * > ( dim, nullptr );
        C_max = A_Vector<mreal * > ( dim, nullptr );
        for( mint k = 0; k < dim; ++ k )
        {
            safe_alloc( C_coords[k], cluster_count, 0. );
            safe_alloc( C_min[k], cluster_count );
            safe_alloc( C_max[k], cluster_count );
        }
        
        safe_alloc( C_squared_radius, cluster_count );
        
//        C_moments = A_Vector<mreal * restrict> ( moment_count, nullptr );
//        for( mint k = 0; k < moment_count; ++ k )
//        {
//            safe_alloc( C_moments[k], cluster_count, 0. );
//        }
        
        C_D_far = A_Vector<A_Vector<mreal>> ( thread_count );
        #pragma omp parallel for shared( thread_count ) schedule( static, 1 )
        for( mint thread = 0; thread < thread_count; ++thread )
        {
            C_D_far[thread] = A_Vector<mreal> ( cluster_count * far_dim );
        }
        
        // using the already serialized cluster tree
        #pragma omp parallel shared( thread_count )
        {
            #pragma omp single nowait
            {
                computeClusterData( 0, thread_count );
            }
        }
        
        ptoc("OptimizedClusterTree::ComputeClusterData");
    }; //ComputeClusterData


    void OptimizedClusterTree::computeClusterData( const mint C, const mint free_thread_count ) // helper function for ComputeClusterData
    {
        
        mint thread = omp_get_thread_num();
        mint L = C_left [C];
        mint R = C_right[C];
        
        if( L >= 0 && R >= 0 ){
            //C points to interior node.

            #pragma omp task final(free_thread_count<1)  shared( L, free_thread_count ) //firstprivate(free_thread_count)
            {
                computeClusterData( L, free_thread_count/2 );
            }
            #pragma omp task final(free_thread_count<1)  shared( R, free_thread_count )// firstprivate(free_thread_count)
            {
                computeClusterData( R, free_thread_count-free_thread_count/2 );
            }
            #pragma omp taskwait

            //weight
            mreal L_weight = C_far[0][L];
            mreal R_weight = C_far[0][R];
            mreal C_mass = L_weight + R_weight;
            C_far[0][C] = C_mass;
            
            mreal C_invmass = 1./C_mass;
            L_weight = L_weight * C_invmass;
            R_weight = R_weight * C_invmass;
        
            for( mint k = 1, last = far_dim; k < last; ++k )
            {
                C_far[k][C] = L_weight * C_far[k][L]  + R_weight * C_far[k][R] ;
            }
            //clustering coordinates and bounding boxes
            for( mint k = 0, last = dim; k < last; ++k )
            {
                C_coords[k][C] = L_weight * C_coords[k][L] + R_weight * C_coords[k][R];
                C_min[k][C] = mymin( C_min[k][L], C_min[k][R] );
                C_max[k][C] = mymax( C_max[k][L], C_max[k][R] );
            }
            
//            ShiftMoments( L, C_far, C_moments, C, C_far, C_moments, &scratch[thread][0] );
//            ShiftMoments( R, C_far, C_moments, C, C_far, C_moments, &scratch[thread][0] );
        }
        else
        {
            //C points to leaf node.
            //compute from primitives
            
            mint begin = C_begin[C];
            mint end   = C_end  [C];
            // Mass
            mreal C_mass = 0.;
            for( mint i = begin; i < end; ++i )
            {
                C_mass += P_far[0][i];
            }
            C_far[0][C] = C_mass;
            mreal C_invmass = 1./C_mass;
            
            
            // weighting the coordinates
            for( mint i = begin; i < end; ++i )
            {
                mreal P_weight = P_far[0][i] * C_invmass;
                for( mint k = 1; k < far_dim; ++k )
                {
                    C_far[k][C] += P_weight * P_far[k][i];
                }
                for( mint k = 0; k < dim; ++k )
                {
                    C_coords[k][C] += P_weight * P_coords[k][i];
                }
            }
            
//            // moments
//            for( mint i = begin; i < end; ++i )
//            {
//                ShiftMoments( i, P_far, P_moments, C, C_far, C_moments, &scratch[thread][0] );
//            }
            
            // bounding boxes
            for( mint k = 0; k < dim; ++ k )
            {
                C_min[k][C] = *std::min_element( P_min[k] + begin, P_min[k] + end  );
                C_max[k][C] = *std::max_element( P_max[k] + begin, P_max[k] + end  );
            }
            

        }
        
        // finally, we compute the square radius of the bounding box, measured from the clusters barycenter C_coords
        mreal r2 = 0.;
        for( mint k = 0; k < dim; ++k )
        {
            mreal mid = C_coords[k][C];
            mreal delta_max = abs( C_max[k][C] - mid );
            mreal delta_min = abs( mid - C_min[k][C] );
            r2 += (delta_min <= delta_max) ?  delta_max * delta_max :  delta_min * delta_min;
        }
        C_squared_radius[C] = r2;
    }; //computeClusterData

    void OptimizedClusterTree::ComputePrePost( MKLSparseMatrix & DiffOp, MKLSparseMatrix & AvOp )
    {
    //    tic("Create pre and post");

        P_to_C = MKLSparseMatrix( cluster_count, primitive_count, primitive_count );
        P_to_C.outer[0] = 0;

        C_to_P = MKLSparseMatrix(primitive_count, cluster_count, primitive_count );
        C_to_P.outer[primitive_count] = primitive_count;
        

        safe_alloc( leaf_cluster_ptr, leaf_cluster_count + 1  );
        leaf_cluster_ptr[0] = 0;
    //    P_leaf = A_Vector<mint>( primitive_count );

        #pragma omp parallel for
        for( mint i = 0; i < leaf_cluster_count; ++i )
        {
            mint leaf = leaf_clusters[i];
            mint begin = C_begin[leaf];
            mint end   = C_end  [leaf];
            leaf_cluster_ptr[ i + 1 ] = end;
            for( mint k = begin; k < end; ++k )
            {
                C_to_P.inner[k] = leaf;
            }
        }

        {
            mreal * x = C_to_P.values;
            mreal * y = P_to_C.values;
            mint * i  = C_to_P.outer;
            mint * j  = P_to_C.inner;
            #pragma omp simd aligned ( x, y, i, j  : ALIGN )
            for( mint k = 0; k < primitive_count; ++k )
            {
                x[k] = 1.;
                y[k] = 1.;
                
                i[k] = k;
                j[k] = k;
            }
        }

        for (mint C = 0; C < cluster_count; ++C)
        {
            if( C_left[C] >= 0)
            {
                P_to_C.outer[C + 1] = P_to_C.outer[C];
            }
            else
            {
                P_to_C.outer[C + 1] = P_to_C.outer[C] + C_end[C] - C_begin[C];
            }
        }

        auto hi_perm = MKLSparseMatrix( dim * primitive_count, dim * primitive_count, dim * primitive_count );
        hi_perm.outer[ dim * primitive_count ] = dim * primitive_count;

        #pragma omp parallel for
        for( mint i = 0; i < primitive_count; ++i )
        {
            mreal a = P_near[0][i];
            for( mint k = 0; k < dim; ++k )
            {
                mint to = dim * i + k;
                hi_perm.outer [ to ] = to;
                hi_perm.inner [ to ] = dim * P_ext_pos[i] + k;
                hi_perm.values[ to ] = a;
            }
        }

        hi_perm.Multiply( DiffOp, hi_pre );

        hi_pre.Transpose( hi_post );

        auto lo_perm = MKLSparseMatrix( primitive_count, primitive_count, C_to_P.outer, P_ext_pos, P_near[0] ); // Copy

        lo_perm.Multiply( AvOp, lo_pre );

        lo_pre.Transpose( lo_post );
    } // ComputePrePost
    
//    void OptimizedClusterTree::ComputePrePost( MKLSparseMatrix & DiffOp, MKLSparseMatrix & AvOp )
//    {
//        ptic("OptimizedClusterTree::ComputePrePost");
//
//        P_to_C = MKLSparseMatrix( cluster_count, primitive_count, primitive_count );
//        P_to_C.outer[0] = 0;
//
//        C_to_P = MKLSparseMatrix(primitive_count, cluster_count, primitive_count );
//        C_to_P.outer[primitive_count] = primitive_count;
//        
//        safe_alloc( leaf_cluster_ptr, leaf_cluster_count + 1  );
//        leaf_cluster_ptr[0] = 0;
//
//        #pragma omp parallel for
//        for( mint i = 0; i < leaf_cluster_count; ++i )
//        {
//            mint leaf = leaf_clusters[i];
//            mint begin = C_begin[leaf];
//            mint end   = C_end  [leaf];
//            leaf_cluster_ptr[ i + 1 ] = end;
//            for( mint k = begin; k < end; ++k )
//            {
//                C_to_P.inner[k] = leaf;
//            }
//        }
//        
//        {
//            mreal * x = C_to_P.values;
//            mint * i  = C_to_P.outer;
//            mreal * y = P_to_C.values;
//            mint * j  = P_to_C.inner;
//            #pragma omp parallel for
//            for( mint k = 0; k < primitive_count; ++k )
//            {
//                x[k] = 1.;
//                y[k] = 1.;
//                i[k] = k;
//                j[k] = k;
//            }
//        }
//        
//        for (mint C = 0; C < cluster_count; ++C)
//        {
//            if( C_left[C] >= 0)
//            {
//                P_to_C.outer[C + 1] = P_to_C.outer[C];
//            }
//            else
//            {
//                P_to_C.outer[C + 1] = P_to_C.outer[C] + C_end[C] - C_begin[C];
//            }
//        }
//        
//        if( use_old_prepost )
//        {
////            print("hi_pre old");
//            auto hi_perm = MKLSparseMatrix( dim * primitive_count, dim * primitive_count, dim * primitive_count );
//            hi_perm.outer[ dim * primitive_count ] = dim * primitive_count;
//
//            #pragma omp parallel for
//            for( mint i = 0; i < primitive_count; ++i )
//            {
//                mreal a = P_near[0][i];
//                for( mint k = 0; k < dim; ++k )
//                {
//                    mint to = dim * i + k;
//                    hi_perm.outer [ to ] = to;
//                    hi_perm.inner [ to ] = dim * P_ext_pos[i] + k;
//                    hi_perm.values[ to ] = a;
//                }
//            }
//
//            hi_perm.Multiply( DiffOp, hi_pre );
//        }
//        else
//        {
////            print("hi_pre new");
//            hi_pre = MKLSparseMatrix( DiffOp.m, DiffOp.n, DiffOp.nnz );
//            mint * Douter = DiffOp.outer;
//            mint * Dinner = DiffOp.inner;
//            mreal * Dvalues = DiffOp.values;
//            mint * Pouter = hi_pre.outer;
//            mint * Pinner = hi_pre.inner;
//            mreal * Pvalues = hi_pre.values;
//
//            // permuting block rows of DiffOp (dim rows per block row)
//            #pragma omp parallel for
//            for( mint i = 0; i < primitive_count; ++i)
//            {
//                mint j = P_ext_pos[i];
//                mint from = dim * j;
//                mint to = dim * i;
//                #pragma omp simd aligned( Pouter, Douter : ALIGN )
//                for( mint k = 0; k < dim; ++k)
//                {
//                    Pouter[to + k + 1] = Douter[from + k + 1] - Douter[from + k];
//                }
//            }
//            partial_sum( Pouter, Pouter + hi_pre.m + 1);
//
//            #pragma omp parallel for
//            for( mint i = 0; i < primitive_count; ++i)
//            {
//                mreal a = P_far[0][i];
//                mint j = P_ext_pos[i];
//                mint from = Douter[dim * j];
//                mint last = Douter[dim * (j+1)] - from;
//                mint to = Douter[dim * i];
//
//                #pragma omp simd aligned( Dinner, Dvalues, Pinner, Pvalues : ALIGN )
//                for( mint k = 0; k < last; ++k )
//                {
//                    Pinner[to + k] = Dinner[from + k];
//                    Pvalues[to + k] = a * Dvalues[from + k];
//                }
//            }
//        }
//
//        hi_pre.Transpose( hi_post );
//                
//        if( use_old_prepost )
//        {
////            print("lo_pre old");
//            auto lo_perm = MKLSparseMatrix( primitive_count, primitive_count, C_to_P.outer, P_ext_pos, P_near[0] ); // Copy
//
//            lo_perm.Multiply( AvOp, lo_pre );
//
//        }
//        else
//        {
////            print("lo_pre new");
//            
//            lo_pre = MKLSparseMatrix( AvOp.m, AvOp.n, AvOp.nnz );
//            mint * Douter = AvOp.outer;
//            mint * Dinner = AvOp.inner;
//            mreal * Dvalues = AvOp.values;
//            mint * Pouter = lo_pre.outer;
//            mint * Pinner = lo_pre.inner;
//            mreal * Pvalues = lo_pre.values;
//
//            // permuting rows of AvOps
//            #pragma omp parallel for
//            for( mint i = 0; i < primitive_count; ++i)
//            {
//                mint j = P_ext_pos[i];
//                mint from = j;
//                mint to = i;
//                Pouter[to + 1] = Douter[from + 1] - Douter[from ];
//            }
//        
//            std::partial_sum( lo_pre.outer, lo_pre.outer + lo_pre.m + 1, lo_pre.outer );
//            
//            #pragma omp parallel for
//            for( mint i = 0; i < primitive_count; ++i)
//            {
//                mreal a = P_far[0][i];
//                mint j = P_ext_pos[i];
//                mint from = Douter[j];
//                mint last = Douter[j+1] - from;
//                mint to = Douter[i];
//
//                #pragma omp simd aligned( Dinner, Dvalues, Pinner, Pvalues : ALIGN )
//                for( mint k = 0; k < last; ++k )
//                {
//                    Pinner[to + k] = Dinner[from + k];
//                    Pvalues[to + k] = a * Dvalues[from + k];
//                }
//            }
//        }
//        
//        lo_pre.Transpose( lo_post );
//        
//        ptoc("OptimizedClusterTree::ComputePrePost");
//    } // ComputePrePost
    
    void OptimizedClusterTree::RequireBuffers( const mint cols )
    {
        ptic("RequireBuffers");
        // TODO: parallelize allocation
        if( cols > max_buffer_dim )
        {
    //        print("Reallocating buffers to max_buffer_dim = " + std::to_string(cols) + "." );
            max_buffer_dim = cols;

            safe_alloc( P_in, primitive_count * max_buffer_dim, 0. );

            safe_alloc( P_out, primitive_count * max_buffer_dim, 0. );
            
            safe_alloc( C_in, cluster_count * max_buffer_dim, 0. );
            
            safe_alloc( C_out, cluster_count * max_buffer_dim, 0. );
            
        }
        buffer_dim = cols;
        ptoc("RequireBuffers");
        
    }; // RequireBuffers

    void OptimizedClusterTree::CleanseBuffers()
    {
        ptic("CleanseBuffers");
        #pragma omp parallel for simd aligned( P_in, P_out : ALIGN )
        for( mint i = 0; i < primitive_count * buffer_dim; ++i )
        {
            P_in[i] = 0.;
            P_out[i] = 0.;
        }
        
        #pragma omp parallel for simd aligned( C_in, C_out : ALIGN )
        for( mint i = 0; i < cluster_count * buffer_dim; ++i )
        {
            C_in[i] = 0.;
            C_out[i] = 0.;
        }
        ptoc("CleanseBuffers");
    }; // CleanseBuffers

    void OptimizedClusterTree::CleanseD()
    {
        ptic("CleanseD");
        #pragma omp parallel
        {
            mint thread = omp_get_thread_num();
            
            mreal * P = &P_D_near[thread][0];
            mreal * Q = &P_D_far[thread][0];
            mreal * C = &C_D_far[thread][0];
            
            #pragma omp simd aligned( P : ALIGN )
            for( mint i = 0; i < primitive_count * near_dim; ++i )
            {
                P[i] = 0.;
            }
            #pragma omp simd aligned( C : ALIGN )
            for( mint i = 0; i < primitive_count * far_dim; ++i )
            {
                Q[i] = 0.;
            }
            #pragma omp simd aligned( C : ALIGN )
            for( mint i = 0; i < cluster_count * far_dim; ++i )
            {
                C[i] = 0.;
            }
        }
        ptoc("CleanseD");
    }; // CleanseD

    void OptimizedClusterTree::PercolateUp()
    {
        ptic("PercolateUp");
        switch (tree_perc_alg) {
            case TreePercolationAlgorithm::Chunks :
//                print("Using Chunks for percolation");
                PercolateUp_Chunks();
                break;
            case TreePercolationAlgorithm::Tasks :
//                print("Using Tasks for percolation");
                #pragma omp parallel
                {
                    #pragma omp single nowait
                    {
                        PercolateUp_Tasks( 0, thread_count );
                    }
                }
                break;
            case TreePercolationAlgorithm::Sequential :
//                print("Using Sequential for percolation");
                PercolateUp_Seq( 0 );
                break;
                
            default:
//                print("Using Tasks for percolation");
                #pragma omp parallel
                {
                    #pragma omp single nowait
                    {
                        PercolateUp_Tasks( 0, thread_count );
                    }
                }
        }
        ptoc("PercolateUp");
    }; // PercolateUp
    
    void OptimizedClusterTree::PercolateDown()
    {
        ptic("PercolateDown");
        switch (tree_perc_alg) {
            case TreePercolationAlgorithm::Chunks :
//                print("Using Chunks");
                PercolateDown_Chunks();
                break;
            case TreePercolationAlgorithm::Tasks :
//                print("Using Tasks");
                #pragma omp parallel
                {
                    #pragma omp single nowait
                    {
                        PercolateDown_Tasks( 0, thread_count );
                    }
                }
                break;
            case TreePercolationAlgorithm::Sequential :
//                print("Using Sequential");
                PercolateDown_Seq( 0 );
                break;
                
            default:
//                print("Using Tasks");
                #pragma omp parallel
                {
                    #pragma omp single nowait
                    {
                        PercolateDown_Tasks( 0, thread_count );
                    }
                }
        }
        ptoc("PercolateDown");
    }; // PercolateUp
    
    void OptimizedClusterTree::RequireChunks()
    {
        
        ptic("RequireChunks");
        if( !chunks_prepared )
        {
            //TODO: This partitioning strategy may cause that some threads will stay idle during the percolation passes. This is caused by our requirement that each chunks is at least as long as cache line in order to mend false sharing. Should happen only for really cause meshes for which parallelization won't scale anyways.
            
            mint chunk_size = CACHE_LINE_LENGHT * ( (cluster_count + thread_count * CACHE_LINE_LENGHT - 1)  / ( thread_count * CACHE_LINE_LENGHT ) );
            
            chunk_roots = A_Vector<A_Vector<mint>> ( thread_count );

            safe_alloc( C_is_chunk_root, cluster_count, false );
            
            #pragma omp parallel num_threads(thread_count)
            {
                mint thread = omp_get_thread_num();
                
                chunk_roots[thread].reserve( tree_max_depth + 1 );
                
                // last is supposed to be the first position _after_ the chunk belonging to thread thread.
                mint last = std::min( cluster_count, chunk_size * ( thread + 1) );
                
                // first cluster in chunk
                mint C = chunk_size * thread;
                
                // C_next[C]-1 is the last cluster in the subtree with root C.
                // The cluster C is "good" w.r.t. the thread, if and only if it is contained in the chunk, if and only if ( C_next[C] - 1 <= last - 1).
                while( ( 0 <= C < cluster_count ) && (C_next[C] < last) ) // ensure that we do not reference past the end of C_next.
                {
                    chunk_roots[thread].push_back(C);
                    C = C_next[C];
                }
                if( C_next[C] == last )
                {
                    // subtree of last C fits tightly into chunk
                    chunk_roots[thread].push_back(C);
                }
                else
                {
                    // subtree of last C does not fit into chunk; use a breadth-first scan to find the max subtrees of C that do fit into chunk.
                    requireChunks(C, last, thread);
                }
            }
            

            // prepare the tip;
                    
//            std::string f = "/Users/Henrik/Shared/Chunks.tsv";
//            std::ofstream os;
//            std::cout << "Writing chunks to " << f << "." << std::endl;
//            os.open(f);
            

            // this is probably fastest in sequential mode; complexity should be bounded by 2 * tree_max_depth * thread_count.
            for( mint thread = 0; thread < thread_count; ++ thread)
            {
                mint * restrict ptr = &chunk_roots[thread][0];
                mint n = chunk_roots[thread].size();

                for( mint i = 0; i < n; ++i )
                {
//                    os << ptr[i] << "\t ";
                    C_is_chunk_root[ptr[i]] = true;
                }
            }
//            os.close();
            
            chunks_prepared = true;
        }

        ptoc("RequireChunks");
    } // RequireChunks
    
    bool OptimizedClusterTree::requireChunks( mint C, mint last, mint thread)
    {
        // last is supposed to be the first position _after_ the chunk belonging to thread thread.
        // C_next[C]-1 is the last cluster in the subtree with root C.
        // The cluster C is "good" w.r.t. the thread, if and only if it is contained in the chunk, if and only if ( C_next[C] - 1 <= last - 1).
        bool C_good = C_next[C] <= last;
        if( C_good )
        {
            chunk_roots[thread].push_back(C);
        }
        else
        {
            if( (C_left[C] >= 0) && (C_right[C] >= 0) )
            {
                bool left_good = requireChunks( C_left[C], last, thread );
                // If the left subtree is not contained in the chunk, then the right one won't either.
                if( left_good )
                {
                    requireChunks( C_right[C], last, thread );
                }
            }
        }
        return C_good;
    } // requireChunks
    
    void OptimizedClusterTree::PercolateUp_Chunks()
    {
        RequireChunks();
        
        #pragma omp parallel for num_threads(thread_count) schedule( static, 1)
        for( mint thread = 0; thread < thread_count; ++thread )
        {
            for( const auto & C: chunk_roots[thread] ) {
                
                // Takes the subtree of each chunk root and does the percolation.
                PercolateUp_Seq( C );
            }
        }
        
        // Now the chunk roots and everything below them is already updated. We only have to process the tip of the tree. We do it sequentially, treating the chunk roots now as the leaves of the tree:
        percolateUp_Tip( 0  );
        
        // do the tip later
    }; // PercolateUp_Chunks
    
    void OptimizedClusterTree::percolateUp_Tip( const mint C  )
    {
        // C = cluster index
        
        mint L = C_left [C];
        mint R = C_right[C];
        
        if( (!C_is_chunk_root[C]) && (L >= 0) && (R >= 0) )
        {
            // If not a leaf, compute the values of the children first.
            percolateUp_Tip(L);
            percolateUp_Tip(R);
            
            // Aftwards, compute the sum of the two children.
            
            #pragma omp simd aligned( C_in : ALIGN )
            for( mint k = 0; k < buffer_dim; ++k )
            {
                // Overwrite, not add-into. Thus cleansing is not required.
                C_in[ buffer_dim * C + k ] = C_in[ buffer_dim * L + k ] + C_in[ buffer_dim * R + k ];
            }
        }
        
    } // percolateUp_Tip


    void OptimizedClusterTree::PercolateDown_Chunks()
    {
        RequireChunks();
        
        // Treats the chunk roots of the tree as leaves and does a sequential downward percolation.
        percolateDown_Tip( 0 );
        
        // Now the chunk roots and everything above is updated. We can now process everything below the chunk roots in parallel.
    
        #pragma omp parallel for num_threads(thread_count) schedule( static, 1)
        for( mint thread = 0; thread < thread_count; ++thread )
        {
            for( const auto & C: chunk_roots[thread] ) {
                // Takes the subtree of each chunk root and does the percolation.
                PercolateDown_Seq( C );
            }
        }
    }; // PercolateDown_Chunks
    
    void OptimizedClusterTree::percolateDown_Tip( const mint C )
    {
        // C = cluster index
        if( !C_is_chunk_root[C] )
        {
            mint L = C_left [C];
            mint R = C_right[C];
            
            if( ( L >= 0 ) && ( R >= 0 ) )
            {
                #pragma omp simd aligned( C_out : ALIGN )
                for( mint k = 0; k < buffer_dim; ++k )
                {
                    mreal buffer = C_out[ buffer_dim * C + k ];
                    C_out[ buffer_dim * L + k ] += buffer;
                    C_out[ buffer_dim * R + k ] += buffer;
                }
                percolateDown_Tip(L);
                percolateDown_Tip(R);
            }
        }
    }; // percolateDown_Tip
    
    void OptimizedClusterTree::PercolateUp_Seq( const mint C )
    {
        // C = cluster index
        
        mint L = C_left [C];
        mint R = C_right[C];
        
        if( (L >= 0) && (R >= 0) )
        {
            // If not a leaf, compute the values of the children first.
            PercolateUp_Seq(L);
            PercolateUp_Seq(R);
            
            // Aftwards, compute the sum of the two children.
            
            #pragma omp simd aligned( C_in : ALIGN )
            for( mint k = 0; k < buffer_dim; ++k )
            {
                // Overwrite, not add-into. Thus cleansing is not required.
                C_in[ buffer_dim * C + k ] = C_in[ buffer_dim * L + k ] + C_in[ buffer_dim * R + k ];
            }
        }
        
    }; // PercolateUp_Seq


    void OptimizedClusterTree::PercolateDown_Seq(const mint C)
    {
        // C = cluster index
        
        mint L = C_left [C];
        mint R = C_right[C];
        
        if( ( L >= 0 ) && ( R >= 0 ) )
        {
            #pragma omp simd aligned( C_out : ALIGN )
            for( mint k = 0; k < buffer_dim; ++k )
            {
                mreal buffer = C_out[ buffer_dim * C + k ];
                C_out[ buffer_dim * L + k ] += buffer;
                C_out[ buffer_dim * R + k ] += buffer;
            }
            PercolateDown_Seq(L);
            PercolateDown_Seq(R);
        }
    }; // PercolateDown_Seq
    
    void OptimizedClusterTree::PercolateUp_Tasks( const mint C, const mint free_thread_count )
    {
        // C = cluster index
        
        mint L = C_left [C];
        mint R = C_right[C];
        
        if( (L >= 0) && (R >= 0) )
        {
            // If not a leaf, compute the values of the children first.
            #pragma omp task final(free_thread_count<1)  shared( L, free_thread_count )
                PercolateUp_Tasks( L, free_thread_count/2 );
            #pragma omp task final(free_thread_count<1)  shared( R, free_thread_count )
                PercolateUp_Tasks( R, free_thread_count-free_thread_count/2 );
            #pragma omp taskwait
            
            // Aftwards, compute the sum of the two children.
            
            #pragma omp simd aligned( C_in : ALIGN )
            for( mint k = 0; k < buffer_dim; ++k )
            {
                // Overwrite, not add-into. Thus cleansing is not needed.
                C_in[ buffer_dim * C + k ] = C_in[ buffer_dim * L + k ] + C_in[ buffer_dim * R + k ];
            }
        }
        
    }; // PercolateUp


    void OptimizedClusterTree::PercolateDown_Tasks(const mint C, const mint free_thread_count )
    {
        // C = cluster index
        
        mint L = C_left [C];
        mint R = C_right[C];
        
        if( ( L >= 0 ) && ( R >= 0 ) )
        {
            #pragma omp simd aligned( C_out : ALIGN )
            for( mint k = 0; k < buffer_dim; ++k )
            {
                mreal buffer = C_out[ buffer_dim * C + k ];
                C_out[ buffer_dim * L + k ] += buffer;
                C_out[ buffer_dim * R + k ] += buffer;
            }
            
            #pragma omp task final(free_thread_count<1)  shared( L, free_thread_count )
                PercolateDown_Tasks( L, free_thread_count/2 );
            #pragma omp task final(free_thread_count<1)  shared( R, free_thread_count )
                PercolateDown_Tasks( R, free_thread_count-free_thread_count/2 );
            #pragma omp taskwait
        }
    }; // PercolateDown_Tasks


    void OptimizedClusterTree::Pre( Eigen::MatrixXd & input, BCTKernelType type )
    {
        ptic("OptimizedClusterTree::Pre( Eigen::MatrixXd & input, BCTKernelType type )");
        mint cols = input.cols();
    //    tic("Eigen map + copy");
        EigenMatrixRM input_wrapper = EigenMatrixRM( input );
    //    toc("Eigen map + copy");
        
        Pre( input_wrapper.data(), cols, type );
        
        ptoc("OptimizedClusterTree::Pre( Eigen::MatrixXd & input, BCTKernelType type )");
    }

    void OptimizedClusterTree::Pre( mreal * input, const mint cols, BCTKernelType type )
    {
        ptic("Pre");
        MKLSparseMatrix * pre;
        
        switch (type)
        {
            case BCTKernelType::FractionalOnly:
            {
                pre  = &lo_pre ;
                RequireBuffers( cols );
                break;
            }
            case BCTKernelType::HighOrder:
            {
                pre  = &hi_pre ;
                RequireBuffers( dim * cols );                                   // Beware: The derivative operator increases the number of columns!
                break;
            }
            case BCTKernelType::LowOrder:
            {
                pre  = &lo_pre ;
                RequireBuffers( cols );
                break;
            }
            default:
            {
                eprint("Unknown kernel. Doing no.");
                return;
            }
        }
        
        // Caution: Some magic is going on here high order term...
        ptic("pre->Multiply");
    //     Apply diff/averaging operate, reorder and multiply by weights.
        pre->Multiply( input, P_in, cols );
        ptoc("pre->Multiply");
        
        ptic("P_to_C.Multiply");
        // Accumulate into leaf clusters.
        P_to_C.Multiply( P_in, C_in, buffer_dim );  // Beware: The derivative operator increases the number of columns!
        ptoc("P_to_C.Multiply");
        
        PercolateUp();
    
        ptoc("Pre");
    }; // Pre

    void OptimizedClusterTree::Post( Eigen::MatrixXd & output, BCTKernelType type, bool addToResult )
    {
//        tic("Post Eigen");
        mint cols = output.cols();
        
        EigenMatrixRM output_wrapper ( output.rows(), output.cols() );
        
        Post( output_wrapper.data(), cols, type, false);
        
        if( addToResult )
        {
            output += output_wrapper;        // This also converts back to the requested storage type (row/column major).
        }
        else
        {
            output  = output_wrapper;        // This also converts back to the requested storage type (row/column major).
        }
//        toc("Post Eigen");
    }

    void OptimizedClusterTree::Post( mreal * output, const mint cols, BCTKernelType type, bool addToResult )
    {
        ptic("Post");
        MKLSparseMatrix * post;
        
        mint expected_dim = buffer_dim;
        
        switch (type)
        {
            case BCTKernelType::FractionalOnly:
            {
                post  = &lo_post;
                break;
            }
            case BCTKernelType::HighOrder:
            {
                post  = &hi_post;
                expected_dim /= dim;                        // Beware: The derivative operator increases the number of columns!
                break;
            }
            case BCTKernelType::LowOrder:
            {
                post  = &lo_post;
                break;
            }
            default:
            {
                eprint("Unknown kernel. Doing no.");
                return;
            }
        }
        
        if( expected_dim < cols )
        {
            wprint("Expected number of columns  = " + std::to_string( expected_dim ) + " is smaller than requested number of columns " + std::to_string( cols ) + ". Result is very likely unexpected." );
        }
        
        if( expected_dim > cols )
        {
            wprint("Expected number of columns  = " + std::to_string( expected_dim ) + " is greater than requested number of columns " + std::to_string( cols ) + ". Truncating output. Result is very likely unexpected." );
        }
        
        PercolateDown();
        
        // Add data from leaf clusters into data on primitives
        ptic("C_to_P.Multiply");
        C_to_P.Multiply( C_out, P_out, buffer_dim, true );  // Beware: The derivative operator increases the number of columns!
        ptoc("C_to_P.Multiply");
        
        // Multiply by weights, restore external ordering, and apply transpose of diff/averaging operator.
        ptic("post->Multiply");
        post->Multiply( P_out, output, cols, false );
        ptoc("post->Multiply");
        
        ptoc("Post");
    }; // Post

    void OptimizedClusterTree::CollectDerivatives( mreal * restrict const P_D_near_output )
    {
        ptic("OptimizedClusterTree::CollectDerivatives( mreal * restrict const P_D_near_output )");
        #pragma omp parallel for num_threads( thread_count )
        for( mint i = 0; i < primitive_count; ++i )
        {
            mint j = inverse_ordering[i];
            // I did not find out how this can be really parallelized; P_D_near[thread] seems to be the obstruction.
            //        #pragma omp simd aligned( P_out : ALIGN )
            for( mint k = 0; k < near_dim; ++k )
            {
                mreal acc = 0.;
                for( mint thread = 0; thread < thread_count; ++thread )
                {
                    acc += P_D_near[thread][ near_dim * j + k ];
                }
                P_D_near_output[ near_dim * i + k ] = acc;
            }
        }
        ptoc("OptimizedClusterTree::CollectDerivatives( mreal * restrict const P_D_near_output )");
        
    } // CollectDerivatives
    
    void OptimizedClusterTree::CollectDerivatives( mreal * restrict const P_D_near_output, mreal * restrict const P_D_far_output )
    {
        ptic("OptimizedClusterTree::CollectDerivatives");
        
        #pragma omp parallel for num_threads( thread_count )
        for( mint i = 0; i < primitive_count; ++i )
        {
            mint j = inverse_ordering[i];
            // I did not find out how this can be really parallelized; P_D_near[thread] seems to be the obstruction.
            //        #pragma omp simd aligned( P_out : ALIGN )
            for( mint k = 0; k < near_dim; ++k )
            {
                mreal acc = 0.;
                for( mint thread = 0; thread < thread_count; ++thread )
                {
                    acc += P_D_near[thread][ near_dim * j + k ];
                }
                P_D_near_output[ near_dim * i + k ] = acc;
            }
        }
        //    toc("Accumulate primitive contributions");
        
        RequireBuffers(far_dim);

        //    tic("Accumulate cluster contributions");
        #pragma omp parallel for num_threads( thread_count )
        for( mint i = 0; i < cluster_count; ++i )
        {
            // I did not find out how this can be really parallelized; C_D_data[thread] seems to be the obstruction.
            //        #pragma omp simd aligned( C_out : ALIGN )
            for( mint k = 0; k < far_dim; ++k )
            {
                mreal acc = 0.;
                for( mint thread = 0; thread < thread_count; ++thread )
                {
                    acc += C_D_far[thread][ far_dim * i + k ];
                }
                C_out[ far_dim * i + k ]  = acc;
            }
        }
        //    toc("Accumulate cluster contributions");
        
        ptic("PercolateDown");
        PercolateDown();
        ptoc("PercolateDown");
        
        ptic("C_to_P.Multiply");
        C_to_P.Multiply( C_out, P_out, far_dim, false);
        ptoc("C_to_P.Multiply");
        
        #pragma omp parallel for num_threads( thread_count )
        for( mint i = 0; i < primitive_count; ++i )
        {
            mint j = inverse_ordering[i];
            
            #pragma omp simd aligned( P_out, P_D_far_output : ALIGN )
            for( mint k = 0; k < far_dim; ++k )
            {
                P_D_far_output[ far_dim * i + k ] = P_out[ far_dim * j + k ];
            }
        }
        
        ptoc("OptimizedClusterTree::CollectDerivatives");
        
    } // CollectDerivatives
    
    
    void OptimizedClusterTree::SemiStaticUpdate( const mreal * restrict const P_near_, const mreal * restrict const P_far_ ){
        // Updates only the computational data like primitive/cluster areas, centers of mass and normals. All data related to clustering or multipole acceptance criteria remain are unchanged.
        
        ptic("OptimizedClusterTree::SemiStaticUpdate");
        
        RequireBuffers( far_dim );
        
        #pragma omp parallel for shared( P_near, P_far , P_ext_pos, P_near_, P_far_, P_in, near_dim, far_dim )
        for( mint i = 0; i < primitive_count; ++i )
        {
            mint j = P_ext_pos[i];
            
            for( mint k = 0; k < near_dim; ++k )
            {
                P_near[k][i] = P_near_[ near_dim * j + k];
            }
            
            for( mint k = 0; k < far_dim; ++k )
            {
                P_far[k][i] = P_far_[ far_dim * j + k];
            }
            
            //store 0-th moments in the primitive input buffer so that we can use P_to_C and PercolateUp.
            mreal a = P_far_[ far_dim * j];
            P_in[ far_dim * i] = a;
            for( mint k = 1; k < far_dim; ++k )
            {
                P_in[far_dim * i + k] = a * P_far_[far_dim * j + k];
            }
        }
        
        // accumulate primitive input buffers in leaf clusters
        ptic("P_to_C.Multiply");
        P_to_C.Multiply(P_in, C_in, far_dim);
        ptoc("P_to_C.Multiply");
        
        ptic("PercolateUp");
        // upward pass, obviously
        PercolateUp();
        ptoc("PercolateUp");
        
        // finally divide center and normal moments by area and store the result in C_far
        #pragma omp parallel for shared( C_far, C_in, far_dim )
        for( mint i = 0; i < cluster_count; ++i )
        {
            mreal a = C_in[ far_dim * i];
            C_far[0][i] = a;
            mreal ainv = 1./a;
            for( mint k = 1; k < far_dim; ++k )
            {
                C_far[k][i] = ainv * C_in[ far_dim * i + k];
            }
        }
        
        ptoc("OptimizedClusterTree::SemiStaticUpdate");
        
    } // SemiStaticUpdate
    
    
    
    void OptimizedClusterTree::PrintToFile(std::string filename)
    {
        std::ofstream os;
        std::cout << "Writing tree to " << filename << "." << std::endl;
        os.open(filename);
        os << "ID" << "\t" << "left" << "\t" << "right" << "\t" << "next" << "\t" << "depth" << "\t"  << "begin" << "\t" << "end" << std::endl;
        for( mint C = 0;  C < cluster_count; ++C)
        {
            os << C << "\t" << C_left[C] << "\t" << C_right[C] << "\t" << C_next[C] << "\t" << C_depth[C] << "\t" << C_begin[C] << "\t" << C_end[C] << std::endl;
        }
        os.close();
        std::cout << "Done writing." << std::endl;
    }
} // namespace rsurfaces
