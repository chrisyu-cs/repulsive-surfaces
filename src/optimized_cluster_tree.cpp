#include "optimized_cluster_tree.h"

namespace rsurfaces
{
    Cluster2::Cluster2(mint begin_, mint end_, mint depth_)
    {
        begin = begin_;
        end = end_;
        depth = depth_;
        descendant_count = 0;
        descendant_leaf_count = 0;
        left = nullptr;
        right = nullptr;
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
       const mint max_buffer_dim_,
       const mint * restrict const ordering_, // A suggested preordering of primitives; this gets applied before the clustering begins in the hope that this may improve the sorting within a cluster --- at least in the top level(s). This could, e.g., be the ordering obtained by a tree for  similar data set.
       const mint split_threshold_,          // split a cluster if has this many or more primitives contained in it
       MKLSparseMatrix &DiffOp,              // Asking now for MKLSparseMatrix instead of EigenMatrixCSR as input
       MKLSparseMatrix &AvOp                 // Asking now for MKLSparseMatrix instead of EigenMatrixCSR as input
    )
    {
        primitive_count = primitive_count_;
        hull_count = hull_count_;
        dim = dim_;
        near_dim = near_dim_;
        far_dim = far_dim_;
//        moment_count = moment_count_;
        max_buffer_dim = 0;

//        scratch_size = 12;

        mint nthreads;
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }

        tree_thread_count = std::max( static_cast<mint>(1), nthreads );
             thread_count = std::max( static_cast<mint>(1), nthreads );
        
        mint a = 1;
        split_threshold = std::max( a, split_threshold_);
    

        P_coords = A_Vector<mreal * >( dim, nullptr );
        #pragma omp parallel for
        for( mint k = 0; k < dim; ++k)
        {
            P_coords[k] = mreal_alloc(primitive_count);
        }

        P_ext_pos = mint_alloc( primitive_count );
        
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
        
        Cluster2 * root = new Cluster2 ( 0, primitive_count, 0 );

        #pragma omp parallel num_threads(tree_thread_count)  shared( root, P_coords, P_ext_pos, tree_thread_count)
        {
            #pragma omp single
            {
                SplitCluster( root, tree_thread_count );
            }
        }

        cluster_count = root->descendant_count;
        leaf_cluster_count = root->descendant_leaf_count;

        // TODO: Create parallel tasks here.
        {
            mint s = std::max( dim * dim, far_dim);
            RequireBuffers( std::max( s, max_buffer_dim_ ) );
        }
        
        C_left  = mint_alloc ( cluster_count );
        C_right = mint_alloc ( cluster_count );
        C_begin = mint_alloc ( cluster_count );
        C_end   = mint_alloc ( cluster_count );
        C_depth = mint_alloc ( cluster_count );
        
        leaf_clusters = mint_alloc( leaf_cluster_count );
        leaf_cluster_lookup = mint_alloc( cluster_count );

        #pragma omp parallel num_threads(tree_thread_count)  shared( root, tree_thread_count )
        {
            #pragma omp single
            {
                Serialize( root, 0, 0, tree_thread_count );
            }
        }

        delete root;

        inverse_ordering = mint_alloc( primitive_count );
        #pragma omp parallel for
        for( mint i = 0; i < primitive_count; ++i )
        {
            inverse_ordering[P_ext_pos[i]] = i;
        }

        ComputePrimitiveData( P_hull_coords_, P_near_, P_far_ );
//        ComputePrimitiveData( P_hull_coords_, P_near_, P_moments_ );
        
        ComputeClusterData();

        ComputePrePost( DiffOp, AvOp );

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

        P_near = A_Vector<mreal * > ( near_dim, nullptr );
        for( mint k = 0; k < near_dim; ++ k )
        {
            P_near[k] = mreal_alloc( primitive_count );
        }
        
        P_far  = A_Vector<mreal * > ( far_dim , nullptr );
        for( mint k = 0; k < far_dim; ++ k )
        {
            P_far [k] = mreal_alloc( primitive_count );
        }
        
        P_min = A_Vector<mreal * > ( dim, nullptr );
        P_max = A_Vector<mreal * > ( dim, nullptr );
        for( mint k = 0; k < dim; ++ k )
        {
            P_min[k] = mreal_alloc( primitive_count );
            P_max[k] = mreal_alloc( primitive_count );
        }
        
        P_D_near = A_Vector<A_Vector<mreal>> ( thread_count );
        P_D_far  = A_Vector<A_Vector<mreal>> ( thread_count );
        
//        P_moments = A_Vector<mreal * restrict> ( moment_count, nullptr );
//        for( mint k = 0; k < moment_count; ++ k )
//        {
//            P_moments[k] = mreal_alloc( primitive_count );
//        }
            
        mint hull_size = hull_count * dim;
        
        #pragma omp parallel for shared( thread_count ) schedule( static, 1 )
        for( mint thread = 0; thread < thread_count; ++thread )
        {
            P_D_near[thread] = A_Vector<mreal> ( primitive_count * near_dim );
            P_D_far[thread]  = A_Vector<mreal> ( primitive_count * far_dim );
        }
        
        #pragma omp parallel for  shared( P_near, P_far, P_ext_pos, P_min, P_min, P_near_, P_far_, P_hull_coords_, near_dim, far_dim, hull_size, dim )
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
    } //ComputePrimitiveData

    void OptimizedClusterTree::ComputeClusterData()
    {
        
    //    tic("Allocation");
        
//        scratch = A_Vector<A_Vector<mreal>> ( thread_count );
//        for( mint thread = 0; thread < thread_count; ++thread )
//        {
//            scratch[thread] = A_Vector<mreal> ( scratch_size );
//        }
        
        C_far = A_Vector<mreal * > ( far_dim, nullptr );
        for( mint k = 0; k < far_dim; ++ k )
        {
            C_far[k] = mreal_alloc( cluster_count, 0. );
        }
        
        C_coords = A_Vector<mreal * > ( dim, nullptr );
        C_min = A_Vector<mreal * > ( dim, nullptr );
        C_max = A_Vector<mreal * > ( dim, nullptr );
        for( mint k = 0; k < dim; ++ k )
        {
            C_coords[k] = mreal_alloc( cluster_count, 0. );
            C_min[k]    = mreal_alloc( cluster_count );
            C_max[k]    = mreal_alloc( cluster_count );
        }
        
        C_squared_radius = mreal_alloc( cluster_count );
        
//        C_moments = A_Vector<mreal * restrict> ( moment_count, nullptr );
//        for( mint k = 0; k < moment_count; ++ k )
//        {
//            C_moments[k] = mreal_alloc( cluster_count, 0. );
//        }
        
        C_D_far = A_Vector<A_Vector<mreal>> ( thread_count );
        #pragma omp parallel for shared( thread_count ) schedule( static, 1 )
        for( mint thread = 0; thread < thread_count; ++thread )
        {
            C_D_far[thread] = A_Vector<mreal> ( cluster_count * far_dim );
        }
    //    toc("Allocation");
        
        // using the already serialized cluster tree
        #pragma omp parallel  shared( thread_count )
        {
            #pragma omp single
            {
                computeClusterData( 0, thread_count );
            }
        }
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
        

        leaf_cluster_ptr = mint_alloc( leaf_cluster_count + 1  );
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
            mreal a = P_far[0][i];
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

        auto lo_perm = MKLSparseMatrix( primitive_count, primitive_count, C_to_P.outer, P_ext_pos, P_far[0] ); // Copy

        lo_perm.Multiply( AvOp, lo_pre );

        lo_pre.Transpose( lo_post );
    } // ComputePrePost

    void OptimizedClusterTree::RequireBuffers( const mint cols )
    {
        // TODO: parallelize allocation
        if( cols > max_buffer_dim )
        {
    //        print("Reallocating buffers to max_buffer_dim = " + std::to_string(cols) + "." );
            max_buffer_dim = cols;

            mreal_free(P_in);
            P_in = mreal_alloc( primitive_count * max_buffer_dim, 0. );

            mreal_free(P_out);
            P_out = mreal_alloc( primitive_count * max_buffer_dim, 0. );
            
            mreal_free(C_in);
            C_in = mreal_alloc( cluster_count * max_buffer_dim, 0. );
            
            mreal_free(C_out);
            C_out = mreal_alloc( cluster_count * max_buffer_dim, 0. );
            
        }
        
        buffer_dim = cols;
        
    }; // RequireBuffers

    void OptimizedClusterTree::CleanseBuffers()
    {
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
    }; // CleanseBuffers

    void OptimizedClusterTree::CleanseD()
    {
        #pragma omp parallel for schedule(static, 1)
        for( mint thread = 0; thread < thread_count; ++thread )
        {
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
    }; // CleanseD

    void OptimizedClusterTree::PercolateUp( const mint C, const mint free_thread_count )
    {
        // C = cluster index
        
        mint L = C_left [C];
        mint R = C_right[C];
        
        if( (L >= 0) && (R >= 0) )
        {
            // If not a leaf, compute the values of the children first.
            #pragma omp task final(free_thread_count<1)  shared( L, free_thread_count )
                PercolateUp( L, free_thread_count/2 );
            #pragma omp task final(free_thread_count<1)  shared( R, free_thread_count )
                PercolateUp( R, free_thread_count-free_thread_count/2 );
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


    void OptimizedClusterTree::PercolateDown(const mint C, const mint free_thread_count )
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
                PercolateDown( L, free_thread_count/2 );
            #pragma omp task final(free_thread_count<1)  shared( R, free_thread_count )
                PercolateDown( R, free_thread_count-free_thread_count/2 );
            #pragma omp taskwait
        }
    }; // PercolateDown


    void OptimizedClusterTree::Pre( Eigen::MatrixXd & input, BCTKernelType type )
    {
     
        mint cols = input.cols();
    //    tic("Eigen map + copy");
        EigenMatrixRM input_wrapper = EigenMatrixRM( input );
    //    toc("Eigen map + copy");
        
        Pre( input_wrapper.data(), cols, type );
    }

    void OptimizedClusterTree::Pre( mreal * input, const mint cols, BCTKernelType type )
    {
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
    //    tic("MKL pre");
    //     Apply diff/averaging operate, reorder and multiply by weights.
        pre->Multiply( input, P_in, cols );
    //    toc("MKL pre");
        
    //    tic("P_to_C");
        // Accumulate into leaf clusters.
        P_to_C.Multiply( P_in, C_in, buffer_dim );  // Beware: The derivative operator increases the number of columns!
    //    toc("P_to_C");
        
    //    tic("PercolateUp");
        PercolateUp( 0, thread_count );
    //    toc("PercolateUp");
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
//        tic("Post");
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
        
//        tic("PercolateDown");
        PercolateDown( 0, thread_count );
//        toc("PercolateDown");
        
        // Add data from leaf clusters into data on primitives
//        tic("C_to_P");
        C_to_P.Multiply( C_out, P_out, buffer_dim, true );  // Beware: The derivative operator increases the number of columns!
//        toc("C_to_P");
        
        // Multiply by weights, restore external ordering, and apply transpose of diff/averaging operator.
//        tic("MKL post");
        post->Multiply( P_out, output, cols, false );
//        toc("MKL post");
        
//        toc("Post");
    }; // Post

    void OptimizedClusterTree::CollectDerivatives( mreal * restrict const P_D_near_output )
    {
        //    tic("Accumulate primitive contributions");
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
        
    } // CollectDerivatives
    
    void OptimizedClusterTree::CollectDerivatives( mreal * restrict const P_D_near_output, mreal * restrict const P_D_far_output )
    {
        //    tic("Accumulate primitive contributions");
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
        
        //    tic("PercolateDown");
        PercolateDown(0, thread_count );
        //    toc("PercolateDown");
        
        //    tic("C_to_P.");
        C_to_P.Multiply( C_out, P_out, far_dim, false);
        //    toc("C_to_P.");
        
        //    tic("Reorder and copy to output");
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
        //    toc("Reorder and copy to output");
        
    } // CollectDerivatives


void OptimizedClusterTree::SemiStaticUpdate( const mreal * restrict const P_near_, const mreal * restrict const P_far_ ){
    // Updates only the computational data like primitive/cluster areas, centers of mass and normals. All data related to clustering or multipole acceptance criteria remain are unchanged.
    
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
    P_to_C.Multiply(P_in, C_in, far_dim);
    
    // upward pass, obviously
    PercolateUp( 0, thread_count );
    
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
    
} // SemiStaticUpdate

} // namespace rsurfaces
