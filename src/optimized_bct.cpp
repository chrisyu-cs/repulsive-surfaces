#include "optimized_bct.h"

namespace rsurfaces
{

    OptimizedBlockClusterTree::OptimizedBlockClusterTree(OptimizedClusterTree* S_, OptimizedClusterTree* T_, const mreal alpha_, const mreal beta_, const mreal theta_, bool exploit_symmetry_, bool upper_triangular_)
    {
        ptic("OptimizedBlockClusterTree::OptimizedBlockClusterTree");
        S = S_;
        T = T_;
        alpha = alpha_;
        beta = beta_;
        
        theta2 = theta_ * theta_;
        is_symmetric = ( S == T );
        exploit_symmetry = is_symmetric && exploit_symmetry_;
        upper_triangular = is_symmetric && upper_triangular_;
        disableNearField = false;
        metrics_initialized = false;

        // TODO: Abort and throw error when S->dim != T->dim
        dim = std::min(S->dim, T->dim);

        exp_s = (beta - intrinsic_dim) / alpha;
        mreal sigma = exp_s - 1.0;
        hi_exponent = -0.5 * (2.0 * sigma + intrinsic_dim); // multiplying by 0.5 because we use r * r instead of r for saving a sqrt
                                                            //    fr_exponent = - 0.5 * (2.0 * exp_s + intrinsic_dim); // from Chris' code
                                                            //    fr_exponent =  -0.5 * ( 2. (2. - exp_s) + intrinsic_dim); //shouldn't it be this?

        #pragma omp parallel
        {
            thread_count = omp_get_num_threads();
            tree_thread_count = omp_get_num_threads();
        }

        // tic("CreateBlockClusters");
        CreateBlockClusters();
        // toc("CreateBlockClusters");

        // TODO: The following line should be moved to InternalMultiply in order to delay matrix creation to a time when it is actually needed. Otherwise, using the BCT for line search (evaluating only the energy), the time for creating the matrices would be wasted.
        RequireMetrics();

        ptoc("OptimizedBlockClusterTree::OptimizedBlockClusterTree");
    }; // Constructor

    //######################################################################################################################################
    //      Initialization
    //######################################################################################################################################

    void OptimizedBlockClusterTree::CreateBlockClusters()
    {
        ptic("OptimizedBlockClusterTree::CreateBlockClusters");
        auto thread_sep_idx = A_Vector<A_Deque<mint>>(tree_thread_count);
        auto thread_sep_jdx = A_Vector<A_Deque<mint>>(tree_thread_count);

        auto thread_nonsep_idx = A_Vector<A_Deque<mint>>(tree_thread_count);
        auto thread_nonsep_jdx = A_Vector<A_Deque<mint>>(tree_thread_count);

        ptic("SplitBlockCluster");

        #pragma omp parallel num_threads(tree_thread_count) shared(thread_sep_idx, thread_sep_jdx, thread_nonsep_idx, thread_nonsep_jdx)
        {
            #pragma omp single
            {
                SplitBlockCluster(thread_sep_idx, thread_sep_jdx, thread_nonsep_idx, thread_nonsep_jdx, 0, 0, tree_thread_count);
            }
        }

        mint sep_blockcluster_count = 0;
        mint nonsep_blockcluster_count = 0;

        for (mint thread = 0; thread < tree_thread_count; ++thread)
        {
            sep_blockcluster_count += thread_sep_idx[thread].size();
            nonsep_blockcluster_count += thread_nonsep_idx[thread].size();
        }

        ptoc("SplitBlockCluster");

        far  = std::make_shared<InteractionData> ( thread_sep_idx, thread_sep_jdx, S->cluster_count, T->cluster_count, upper_triangular );

        near = std::make_shared<InteractionData> ( thread_nonsep_idx, thread_nonsep_jdx, S->leaf_cluster_count, T->leaf_cluster_count, upper_triangular );

        ptoc("OptimizedBlockClusterTree::CreateBlockClusters");
    }; //CreateBlockClusters

    void OptimizedBlockClusterTree::SplitBlockCluster(
        A_Vector<A_Deque<mint>> &sep_i,
        A_Vector<A_Deque<mint>> &sep_j,
        A_Vector<A_Deque<mint>> &nsep_i,
        A_Vector<A_Deque<mint>> &nsep_j,
        const mint i,
        const mint j,
        const mint free_thread_count
    )
    {
        //    std::pair<mint,mint> minmax;
        mint thread = omp_get_thread_num();

        mreal r2i = S->C_squared_radius[i];
        mreal r2j = T->C_squared_radius[j];
        mreal h2 = std::max(r2i, r2j);

        // Compute squared distance between bounding boxes.
        // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares
        mreal R2 = 0.;
        mreal dk = 0.;
        for (mint k = 0; k < dim; ++k)
        {
            mreal xmin = S->C_min[k][i];
            mreal xmax = S->C_max[k][i];

            mreal ymin = T->C_min[k][j];
            mreal ymax = T->C_max[k][j];
            dk = std::max(0., std::max(xmin, ymin) - std::min(xmax, ymax));

            R2 += dk * dk;
        }

        if (h2 > theta2 * R2)
        {

            mint lefti = S->C_left[i];
            mint righti = S->C_right[i];

            mint leftj = T->C_left[j];
            mint rightj = T->C_right[j];

            // Warning: This assumes that either both children are defined or empty.
            if ((lefti >= 0) || (leftj >= 0))
            {

                mreal scorei = (lefti >= 0) ? r2i : 0.;
                mreal scorej = (leftj >= 0) ? r2j : 0.;

                //            mma::print(" scores = ( "+std::to_string(scorei)+" , "+std::to_string(scorej)+" )");

                if (scorei == scorej && scorei > 0. && scorej > 0.)
                {
                    // tie breaker: split both clusters

                    if ((exploit_symmetry) && (i == j))
                    {
                        //                mma::print(" Creating 3 blockcluster children.");
                        mint spawncount = free_thread_count / 3;
                        mint remainder = free_thread_count % 3;

// TODO: These many arguments in the function calls might excert quite a pressure on the stack. Is there a better way to share sep_i, sep_j, nsep_i, nsep_j among all threads other than making them members of the class?
                        #pragma omp task final(free_thread_count < 1) firstprivate(lefti, leftj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, lefti, leftj, spawncount + (remainder > 0));
                        #pragma omp task final(free_thread_count < 1) firstprivate(lefti, rightj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, lefti, rightj, spawncount + (remainder > 2));
                        #pragma omp task final(free_thread_count < 1) firstprivate(righti, rightj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, righti, rightj, spawncount);
                        //                    #pragma omp taskwait
                    }
                    else
                    {
                        // In case of exploit_symmetry !=0, this is a very seldom case; still requird to preserve symmetry.
                        // This happens only if i and j represent _diffent clusters with same radii.

                        mint spawncount = free_thread_count / 4;
                        mint remainder = free_thread_count % 4;

                        #pragma omp task final(free_thread_count < 1) firstprivate(lefti, leftj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, lefti, leftj, spawncount + (remainder > 0));
                        #pragma omp task final(free_thread_count < 1) firstprivate(righti, leftj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, righti, leftj, spawncount + (remainder > 1));
                        #pragma omp task final(free_thread_count < 1) firstprivate(lefti, rightj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, lefti, rightj, spawncount + (remainder > 2));
                        #pragma omp task final(free_thread_count < 1) firstprivate(righti, rightj, spawncount) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, righti, rightj, spawncount);
                        //                    #pragma omp taskwait
                    }
                }
                else
                {
                    // split only larger cluster
                    if (scorei > scorej)
                    {
                        //split cluster i
                        #pragma omp task final(free_thread_count < 1) firstprivate(lefti) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, lefti, j, free_thread_count / 2);
                        #pragma omp task final(free_thread_count < 1) firstprivate(righti) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, righti, j, free_thread_count - free_thread_count / 2);
                        //                    #pragma omp taskwait
                    }
                    else //scorei < scorej
                    {
//split cluster j
                        #pragma omp task final(free_thread_count < 1) firstprivate(leftj) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, i, leftj, free_thread_count / 2);
                        #pragma omp task final(free_thread_count < 1) firstprivate(rightj) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, i, rightj, free_thread_count - free_thread_count / 2);
                        //                    #pragma omp taskwait
                    }
                }
            }
            else
            {
                // create nonsep leaf blockcluster

                // i and j must be leaves of each ClusterTree S and T, so we directly store their position in the list leaf_clusters. This is important for the sparse matrix generation.

                //            In know  this is a very deep branching. I optimized it a bit for the case exploit_symmetry != 0 and upper_triangular == 0, though. That seemed to work best in regard of the matrix-vector multiplication.
                // TODO: Is there a clever way to avoid at least a bit of complixity of this branching? Would that speed up anything in the first place?
                if (exploit_symmetry)
                {
                    if (!upper_triangular)
                    {
                        if (i != j)
                        {
                            // Generate also the twin to get a full matrix.
                            nsep_i[thread].push_back(S->leaf_cluster_lookup[i]);
                            nsep_i[thread].push_back(S->leaf_cluster_lookup[j]);
                            nsep_j[thread].push_back(T->leaf_cluster_lookup[j]);
                            nsep_j[thread].push_back(T->leaf_cluster_lookup[i]);
                        }
                        else
                        {
                            // This is a diagonal block; there is no twin to think about
                            nsep_i[thread].push_back(T->leaf_cluster_lookup[i]);
                            nsep_j[thread].push_back(S->leaf_cluster_lookup[i]);
                        }
                    }
                    else
                    {
                        // For creating an upper triangle matrix we store the pair  { min(i,j), max(i,j) }.
                        if (i <= j)
                        {
                            nsep_i[thread].push_back(S->leaf_cluster_lookup[i]);
                            nsep_j[thread].push_back(T->leaf_cluster_lookup[j]);
                        }
                        else
                        {
                            nsep_i[thread].push_back(T->leaf_cluster_lookup[j]);
                            nsep_j[thread].push_back(S->leaf_cluster_lookup[i]);
                        }
                    }
                }
                else
                {
                    // No symmetry exploited.
                    nsep_i[thread].push_back(S->leaf_cluster_lookup[i]);
                    nsep_j[thread].push_back(T->leaf_cluster_lookup[j]);
                }
            }
        }
        else
        {
            //create sep leaf blockcluster
            if (exploit_symmetry)
            {
                if (!upper_triangular)
                {
                    // Generate also the twin to get a full matrix
                    sep_i[thread].push_back(i);
                    sep_i[thread].push_back(j);
                    sep_j[thread].push_back(j);
                    sep_j[thread].push_back(i);
                }
                else
                {
                    // For creating an upper triangle matrix we store the pair  { min(i,j), max(i,j) }.
                    if (i <= j)
                    {
                        sep_i[thread].push_back(i);
                        sep_j[thread].push_back(j);
                    }
                    else
                    {
                        sep_i[thread].push_back(j);
                        sep_j[thread].push_back(i);
                    }
                }
            }
            else
            {
                // No symmetry exploited.
                sep_i[thread].push_back(i);
                sep_j[thread].push_back(j);
            }
        }
    }; //SplitBlockCluster


    //######################################################################################################################################
    //      Initialization of metrics
    //######################################################################################################################################


    void OptimizedBlockClusterTree::RequireMetrics()
    {
        ptic("OptimizedBlockClusterTree::RequireMetrics");
        if( !metrics_initialized )
        {
            
//            tic("far->PrepareCSR()");
            far->PrepareCSR();
//            toc("far->PrepareCSR()");

            
//            tic("FarFieldInteraction");
            FarFieldInteraction();
//            toc("FarFieldInteraction");

//            tic("near->PrepareCSR( S->leaf_cluster_ptr, T->leaf_cluster_ptr );");
            near->PrepareCSR( S->leaf_cluster_count, S->leaf_cluster_ptr, T->leaf_cluster_count, T->leaf_cluster_ptr );
//            toc("near->PrepareCSR( S->leaf_cluster_ptr, T->leaf_cluster_ptr );");
            
//            tic("NearFieldInteractionCSR");
            NearFieldInteractionCSR();
//            toc("NearFieldInteractionCSR");

//            tic("ComputeDiagonals");
            ComputeDiagonals();
//            toc("ComputeDiagonals");

            metrics_initialized = true;
//          print("Done: RequireMetrics.");
        }
        ptoc("OptimizedBlockClusterTree::RequireMetrics");
    } // RequireMetrics

    void OptimizedBlockClusterTree::FarFieldInteraction()
    {
        ptic("OptimizedBlockClusterTree::FarFieldInteraction");
        mint b_m = far->b_m;
        mint const * restrict const b_outer = far->b_outer;
        mint const * restrict const b_inner = far->b_inner;
        
        // "restrict" makes sense to me here because it exclude write conflicts.
        mreal * restrict const fr_values = far->fr_values;
        mreal * restrict const hi_values = far->hi_values;
        mreal * restrict const lo_values = far->lo_values;

        mreal t1 = intrinsic_dim == 1;
        mreal t2 = intrinsic_dim == 2;
        mreal t3 = intrinsic_dim == 3;
        
        
        if( S->far_dim == 10 && T->far_dim == 10 )
        {
            // Dunno why "restrict" helps with C_far. It is actually a lie here.
            mreal const * restrict const X1 = S->C_far[1];
            mreal const * restrict const X2 = S->C_far[2];
            mreal const * restrict const X3 = S->C_far[3];
            mreal const * restrict const P11 = S->C_far[4];
            mreal const * restrict const P12 = S->C_far[5];
            mreal const * restrict const P13 = S->C_far[6];
            mreal const * restrict const P22 = S->C_far[7];
            mreal const * restrict const P23 = S->C_far[8];
            mreal const * restrict const P33 = S->C_far[9];
            
            mreal const * restrict const Y1 = T->C_far[1];
            mreal const * restrict const Y2 = T->C_far[2];
            mreal const * restrict const Y3 = T->C_far[3];
            mreal const * restrict const Q11 = T->C_far[4];
            mreal const * restrict const Q12 = T->C_far[5];
            mreal const * restrict const Q13 = T->C_far[6];
            mreal const * restrict const Q22 = T->C_far[7];
            mreal const * restrict const Q23 = T->C_far[8];
            mreal const * restrict const Q33 = T->C_far[9];
            
            // Using i and j for cluster positions.
            #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
            for (mint i = 0; i < b_m; ++i)
            {
                mreal x1 = X1[i];
                mreal x2 = X2[i];
                mreal x3 = X3[i];
                mreal p11 = P11[i];
                mreal p12 = P12[i];
                mreal p13 = P13[i];
                mreal p22 = P22[i];
                mreal p23 = P23[i];
                mreal p33 = P33[i];
                
                mint k_begin = b_outer[i];
                mint k_end = b_outer[i+1];
                // This loop can be SIMDized straight-forwardly (horizontal SIMDization).
                // It is in no way the bottleneck at the moment. OptimizedBlockClusterTree::NearFieldEnergy takes many times longer.
                #pragma omp simd aligned( X2, Y1, Y2, Y3, Q11, Q12, Q13, Q22, Q23, Q33 : ALIGN )
                for (mint k = k_begin; k < k_end; ++k)
                {
                    mint j = b_inner[k]; // We are in  block {i, j}
                    
                    mreal v1 = Y1[j] - x1;
                    mreal v2 = Y2[j] - x2;
                    mreal v3 = Y3[j] - x3;
                    mreal q11 = Q11[j];
                    mreal q12 = Q12[j];
                    mreal q13 = Q13[j];
                    mreal q22 = Q22[j];
                    mreal q23 = Q23[j];
                    mreal q33 = Q33[j];
                    
                    mreal rCosPhi2 = v1*(p11*v1 + p12*v2 + p13*v3) + v2*(p12*v1 + p22*v2 + p23*v3) + v3*(p13*v1 + p23*v2 + p33*v3);
                    mreal rCosPsi2 = v1*(q11*v1 + q12*v2 + q13*v3) + v2*(q12*v1 + q22*v2 + q23*v3) + v3*(q13*v1 + q23*v2 + q33*v3);
                    mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 ;
                    
                    mreal hi = mypow(r2, hi_exponent); // I got it down to this single call to pow. We might want to generate a lookup table for it...
                    
                    
                    hi_values[k] = 2.0 * hi; // The factor 2.0 might be suboptimal. That's what my Mathematica code uses and it seems to work fine.
                    
                    //                        near->fr_values[ptr] = pow( r2, fr_exponent );
                    
                    mreal r4 = r2 * r2;
                    mreal r6 = r4 * r2;
                    mreal r8 = r4 * r4;
                    // Nasty trick to enforce vectorization without resorting to mypow or pos. Works only if intrinsic_dim is one of 1, 2, or 3.
                    mreal mul = t1 * r4 + t2 * r6 + t3 * r8;
                    
                    fr_values[k] = 1. / (hi * mul);
                    
                    lo_values[k] = (rCosPhi2 + rCosPsi2) / (r2 * r2) * hi;
                }
            }
        }
        else
        {
            // Dunno why "restrict" helps with C_far. It is actually a lie here.
            mreal const * restrict const X1 = S->C_far[1];
            mreal const * restrict const N1 = S->C_far[4];
            mreal const * restrict const X2 = S->C_far[2];
            mreal const * restrict const N2 = S->C_far[5];
            mreal const * restrict const X3 = S->C_far[3];
            mreal const * restrict const N3 = S->C_far[6];
            
            mreal const * restrict const Y1 = T->C_far[1];
            mreal const * restrict const M1 = T->C_far[4];
            mreal const * restrict const Y2 = T->C_far[2];
            mreal const * restrict const M2 = T->C_far[5];
            mreal const * restrict const Y3 = T->C_far[3];
            mreal const * restrict const M3 = T->C_far[6];
            
            
            // Using i and j for cluster positions.
            #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
            for (mint i = 0; i < b_m; ++i)
            {
                mreal x1 = X1[i];
                mreal x2 = X2[i];
                mreal x3 = X3[i];
                mreal n1 = N1[i];
                mreal n2 = N2[i];
                mreal n3 = N3[i];
                
                mint k_begin = b_outer[i];
                mint k_end = b_outer[i+1];
                // This loop can be SIMDized straight-forwardly (horizontal SIMDization).
                // It is in no way the bottleneck at the moment. OptimizedBlockClusterTree::NearFieldEnergy takes many times longer.
                #pragma omp simd aligned( X2, Y1, Y2, Y3, M1, M2, M3 : ALIGN )
                for (mint k = k_begin; k < k_end; ++k)
                {
                    mint j = b_inner[k]; // We are in  block {i, j}
                    
                    mreal v1 = Y1[j] - x1;
                    mreal v2 = Y2[j] - x2;
                    mreal v3 = Y3[j] - x3;
                    mreal m1 = M1[j];
                    mreal m2 = M2[j];
                    mreal m3 = M3[j];
                    
                    mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                    mreal rCosPsi = v1 * m1 + v2 * m2 + v3 * m3;
                    mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;
                    
                    mreal hi = mypow(r2, hi_exponent); // I got it down to this single call to pow. We might want to generate a lookup table for it...
                    
                    
                    hi_values[k] = 2.0 * hi; // The factor 2.0 might be suboptimal. That's what my Mathematica code uses and it seems to work fine.
                    
                    //                        near->fr_values[ptr] = pow( r2, fr_exponent );
                    
                    mreal r4 = r2 * r2;
                    mreal r6 = r4 * r2;
                    mreal r8 = r4 * r4;
                    // Nasty trick to enforce vectorization without resorting to mypow or pos. Works only if intrinsic_dim is one of 1, 2, or 3.
                    mreal mul = t1 * r4 + t2 * r6 + t3 * r8;
                    
                    fr_values[k] = 1. / (hi * mul);
                    
                    lo_values[k] = (rCosPhi * rCosPhi + rCosPsi * rCosPsi) / (r2 * r2) * hi;
                }
            }
        }
        ptoc("OptimizedBlockClusterTree::FarFieldInteraction");
    }; //FarFieldInteraction

    void OptimizedBlockClusterTree::NearFieldInteractionCSR()
    {
        ptic("OptimizedBlockClusterTree::NearFieldInteractionCSR");
        mint b_m = near->b_m;
        // Getting the pointers first to reduce indexing within the loops. Together with const + restrict, this gains about 5% runtime improvement.
        mint const * restrict const b_row_ptr = &near->b_row_ptr[0];
        mint const * restrict const b_col_ptr = &near->b_col_ptr[0];
        mint const * restrict const b_outer = &near->b_outer[0];
        mint const * restrict const b_inner = &near->b_inner[0];
        mint const * restrict const outer = &near->outer[0];

        // "restrict" makes sense to me here because it exclude write conflicts.
        mreal * restrict const fr_values = &near->fr_values[0];
        mreal * restrict const hi_values = &near->hi_values[0];
        mreal * restrict const lo_values = &near->lo_values[0];

        mreal t1 = intrinsic_dim == 1;
        mreal t2 = intrinsic_dim == 2;
        mreal t3 = intrinsic_dim == 3;

        if( S->near_dim == 10 && T->near_dim == 10 )
        {
            // Dunno why "restrict" helps with P_near. It is actually a lie here if S = S.
            mreal const * restrict const X1 = S->P_near[1];
            mreal const * restrict const X2 = S->P_near[2];
            mreal const * restrict const X3 = S->P_near[3];
            mreal const * restrict const P11 = S->P_near[4];
            mreal const * restrict const P12 = S->P_near[5];
            mreal const * restrict const P13 = S->P_near[6];
            mreal const * restrict const P22 = S->P_near[7];
            mreal const * restrict const P23 = S->P_near[8];
            mreal const * restrict const P33 = S->P_near[9];
            
            mreal const * restrict const Y1 = T->P_near[1];
            mreal const * restrict const Y2 = T->P_near[2];
            mreal const * restrict const Y3 = T->P_near[3];
            mreal const * restrict const Q11 = T->P_near[4];
            mreal const * restrict const Q12 = T->P_near[5];
            mreal const * restrict const Q13 = T->P_near[6];
            mreal const * restrict const Q22 = T->P_near[7];
            mreal const * restrict const Q23 = T->P_near[8];
            mreal const * restrict const Q33 = T->P_near[9];
            
            // Using b_i and b_j for block (leaf cluster) positions.
            // Using i and j for primitive positions.
            #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
            for (mint b_i = 0; b_i < b_m; ++b_i) // we are going to loop over all rows in block fashion
            {
                mint k_begin = b_outer[b_i];
                mint k_end = b_outer[b_i + 1];
                
                mint i_begin = b_row_ptr[b_i];
                mint i_end = b_row_ptr[b_i + 1];
                
                for (mint i = i_begin; i < i_end; ++i) // looping over all rows i  in block row b_i
                {
                    mint ptr = outer[i]; // get first nonzero position in row i; ptr will be used to keep track of the current position within values
                    
                    mreal x1 = X1[i];
                    mreal x2 = X2[i];
                    mreal x3 = X3[i];
                    mreal p11 = P11[i];
                    mreal p12 = P12[i];
                    mreal p13 = P13[i];
                    mreal p22 = P22[i];
                    mreal p23 = P23[i];
                    mreal p33 = P33[i];
                    
                    // From here on, the read-access to T->P_near is a bit cache-unfriendly.
                    // However, the threads write to disjoint large consecutive blocks of memory. Because write is consecutively, false sharing is probably not an issue.
                    
                    for (mint k = k_begin; k < k_end; ++k) // loop over all blocks in block row b_i
                    {
                        
                        mint b_j = b_inner[k]; // we are in block {b_i, b_j} now
                        
                        mint j_begin = b_col_ptr[b_j];
                        mint j_end = b_col_ptr[b_j + 1];
                        
                        #pragma omp simd aligned( Y1, Y2, Y3, Q11, Q12, Q13, Q22, Q23, Q33 : ALIGN )
                        for (mint j = j_begin; j < j_end; ++j)
                        {
                            if (i != j)
                            {
                                mreal v1 = Y1[j] - x1;
                                mreal v2 = Y2[j] - x2;
                                mreal v3 = Y3[j] - x3;
                                mreal q11 = Q11[j];
                                mreal q12 = Q12[j];
                                mreal q13 = Q13[j];
                                mreal q22 = Q22[j];
                                mreal q23 = Q23[j];
                                mreal q33 = Q33[j];
                                
                                mreal rCosPhi2 = v1*(p11*v1 + p12*v2 + p13*v3) + v2*(p12*v1 + p22*v2 + p23*v3) + v3*(p13*v1 + p23*v2 + p33*v3);
                                mreal rCosPsi2 = v1*(q11*v1 + q12*v2 + q13*v3) + v2*(q12*v1 + q22*v2 + q23*v3) + v3*(q13*v1 + q23*v2 + q33*v3);
                                mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 ;
                                
                                
                                // The following line makes up approx 2/3 of this function's runtime! This is whe we avoid pow as much as possible and replace it with mypow.
                                mreal hi = mypow(r2, hi_exponent); // I got it down to this single call to pow. We might want to generate a lookup table for it...
                                
                                hi_values[ptr] = 2.0 * hi; // The factor 2.0 might be suboptimal. That's what my Mathematica code uses (somwwhat accidentally) and it seems to work fine.
                                
                                //                        near->fr_values[ptr] = pow( r2, fr_exponent );
                                
                                
                                //                            mreal mul = mypow( r2 , intrinsic_dim + 1 );
                                
                                mreal r4 = r2 * r2;
                                mreal r6 = r4 * r2;
                                mreal r8 = r4 * r4;
                                // Nasty trick to enforce vectorization without resorting to mypow or pos. Works only if intrinsic_dim is one of 1, 2, or 3.
                                mreal mul = t1 * r4 + t2 * r6 + t3 * r8;
                                
                                fr_values[ptr] = 1. / (hi * mul);
                                
                                lo_values[ptr] = (rCosPhi2 + rCosPsi2) / r4 * hi;
                            }
                            else
                            {
                                // Overwrite diagonal. Just in case.
                                fr_values[ptr] = 0.;
                                hi_values[ptr] = 0.;
                                lo_values[ptr] = 0.;
                            }
                            
                            // Increment ptr, so that the next value is written to the next position.
                            ++ptr;
                        }
                    }
                }
            }
        }
        else
        {
            // Dunno why "restrict" helps with P_near. It is actually a lie here if S = S.
            mreal const * restrict const X1 = &S->P_near[1][0];
            mreal const * restrict const X2 = &S->P_near[2][0];
            mreal const * restrict const X3 = &S->P_near[3][0];
            mreal const * restrict const N1 = &S->P_near[4][0];
            mreal const * restrict const N2 = &S->P_near[5][0];
            mreal const * restrict const N3 = &S->P_near[6][0];
            
            mreal const * restrict const Y1 = &T->P_near[1][0];
            mreal const * restrict const Y2 = &T->P_near[2][0];
            mreal const * restrict const Y3 = &T->P_near[3][0];
            mreal const * restrict const M1 = &T->P_near[4][0];
            mreal const * restrict const M2 = &T->P_near[5][0];
            mreal const * restrict const M3 = &T->P_near[6][0];
            
            // Using b_i and b_j for block (leaf cluster) positions.
            // Using i and j for primitive positions.
            #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
            for (mint b_i = 0; b_i < b_m; ++b_i) // we are going to loop over all rows in block fashion
            {
                mint k_begin = b_outer[b_i];
                mint k_end = b_outer[b_i + 1];
                
                mint i_begin = b_row_ptr[b_i];
                mint i_end = b_row_ptr[b_i + 1];
                
                for (mint i = i_begin; i < i_end; ++i) // looping over all rows i  in block row b_i
                {
                    mint ptr = outer[i]; // get first nonzero position in row i; ptr will be used to keep track of the current position within values
                    
                    mreal x1 = X1[i];
                    mreal x2 = X2[i];
                    mreal x3 = X3[i];
                    mreal n1 = N1[i];
                    mreal n2 = N2[i];
                    mreal n3 = N3[i];
                    
                    // From here on, the read-access to T->P_near is a bit cache-unfriendly.
                    // However, the threads write to disjoint large consecutive blocks of memory. Because write is consecutively, false sharing is probably not an issue.
                    
                    for (mint k = k_begin; k < k_end; ++k) // loop over all blocks in block row b_i
                    {
                        
                        mint b_j = b_inner[k]; // we are in block {b_i, b_j} now
                        
                        mint j_begin = b_col_ptr[b_j];
                        mint j_end = b_col_ptr[b_j + 1];
                        
                        #pragma omp simd aligned( Y1, Y2, Y3, M1, M2, M3 : ALIGN )
                        for (mint j = j_begin; j < j_end; ++j)
                        {
                            if (i != j)
                            {
                                mreal v1 = Y1[j] - x1;
                                mreal v2 = Y2[j] - x2;
                                mreal v3 = Y3[j] - x3;
                                mreal m1 = M1[j];
                                mreal m2 = M2[j];
                                mreal m3 = M3[j];
                                
                                mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                                mreal rCosPsi = v1 * m1 + v2 * m2 + v3 * m3;
                                mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;
                                
                                // The following line makes up approx 2/3 of this function's runtime! This is whe we avoid pow as much as possible and replace it with mypow.
                                mreal hi = mypow(r2, hi_exponent); // I got it down to this single call to pow. We might want to generate a lookup table for it...
                                
                                hi_values[ptr] = 2.0 * hi; // The factor 2.0 might be suboptimal. That's what my Mathematica code uses (somwwhat accidentally) and it seems to work fine.
                                
                                //                        near->fr_values[ptr] = pow( r2, fr_exponent );
                                
                                
                                //                            mreal mul = mypow( r2 , intrinsic_dim + 1 );
                                
                                mreal r4 = r2 * r2;
                                mreal r6 = r4 * r2;
                                mreal r8 = r4 * r4;
                                // Nasty trick to enforce vectorization without resorting to mypow or pos. Works only if intrinsic_dim is one of 1, 2, or 3.
                                mreal mul = t1 * r4 + t2 * r6 + t3 * r8;
                                
                                fr_values[ptr] = 1. / (hi * mul);
                                
                                lo_values[ptr] = (rCosPhi * rCosPhi + rCosPsi * rCosPsi) / r4 * hi;
                            }
                            else
                            {
                                // Overwrite diagonal. Just in case.
                                fr_values[ptr] = 0.;
                                hi_values[ptr] = 0.;
                                lo_values[ptr] = 0.;
                            }
                            
                            // Increment ptr, so that the next value is written to the next position.
                            ++ptr;
                        }
                    }
                }
            }
        }
        ptoc("OptimizedBlockClusterTree::NearFieldInteractionCSR");
    }; //NearFieldInteractionCSR

    //######################################################################################################################################
    //      Vector multiplication
    //######################################################################################################################################

    void OptimizedBlockClusterTree::Multiply(Eigen::VectorXd &input, Eigen::VectorXd &output, BCTKernelType type, bool addToResult) const
    {
        Multiply(input, output, 1, type, addToResult);
    }

    void OptimizedBlockClusterTree::Multiply(Eigen::VectorXd &input, Eigen::VectorXd &output, const mint cols, BCTKernelType type, bool addToResult) const
    {
        ptic("OptimizedBlockClusterTree::Multiply(Eigen::VectorXd &input, Eigen::VectorXd &output, const mint cols, BCTKernelType type, bool addToResult)");
        // Version for vectors of cols-dimensional vectors. Input and out are assumed to be stored in interleave format.
        // E.g., for a list {v1, v2, v3,...} of  cols = 3-vectors, we expect { v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z, ... }

        mint n = T->lo_pre.n; // Expected length for a vector of scalars
        if ((input.size() >= cols * n) && (output.size() >= cols * n))
        {
            // the acual multiplication

            T->Pre(input.data(), cols, type);

            InternalMultiply(type);

            S->Post(output.data(), cols, type, addToResult);

//            // TODO: Henrik wonders why he did this.
//            if (type == BCTKernelType::HighOrder || type == BCTKernelType::LowOrder)
//            {
//                output *= 0.5;
//            }
        }
        else
        {
            if ((input.size() < cols * n))
            {
                eprint(" in OptimizedBlockClusterTree::Multiply: input vector is to short because" + std::to_string(input.size()) + " < " + std::to_string(cols) + " * " + std::to_string(n) + ".");
            }
            if ((output.size() < cols * n))
            {
                eprint(" in OptimizedBlockClusterTree::Multiply: input vector is to short because" + std::to_string(output.size()) + " < " + std::to_string(cols) + " * " + std::to_string(n) + ".");
            }
        }
        
        ptoc("OptimizedBlockClusterTree::Multiply(Eigen::VectorXd &input, Eigen::VectorXd &output, const mint cols, BCTKernelType type, bool addToResult)");
    }

    //######################################################################################################################################
    //      Matrix multiplication
    //######################################################################################################################################

    void OptimizedBlockClusterTree::Multiply(Eigen::MatrixXd &input, Eigen::MatrixXd &output, BCTKernelType type, bool addToResult) const
    {
        ptic("OptimizedBlockClusterTree::Multiply(Eigen::MatrixXd &input, Eigen::MatrixXd &output, BCTKernelType type, bool addToResult)");
        // Top level routine for the user.
        // Optimized for in and output in row major order.

        // toc("ComputeDiagonals");
        //    tic("T->Pre");
        T->Pre(input, type);
        //    toc("T->Pre");

        //    tic("InternalMultiply");
        InternalMultiply(type);
        //    toc("InternalMultiply");

        //    tic("S->Post");
        S->Post(output, type, addToResult);
        //    toc("S->Post");

        ptoc("OptimizedBlockClusterTree::Multiply(Eigen::MatrixXd &input, Eigen::MatrixXd &output, BCTKernelType type, bool addToResult)");
    }; // Multiply

    void OptimizedBlockClusterTree::InternalMultiply(BCTKernelType type) const
    {
        ptic("OptimizedBlockClusterTree::InternalMultiply");
        // TODO: Make it so that RequireMetrics can be called here to initialize the actual matrices only when they are needed.
//        RequireMetrics();

        mreal * diag = NULL;
        mreal * near_values = NULL;
        mreal * far_values = NULL;

        mreal far_factor = 1.;
        mreal near_factor = 1.;

        mint cols = T->buffer_dim;

        S->RequireBuffers(cols); // Tell the S-side what it has to expect.

        // I know, this looks awful... hash tables with keys from BCTKernelType would be nicer.
        switch (type)
        {
        case BCTKernelType::FractionalOnly:
        {
            far_factor = fr_far_factor;
            near_factor = fr_near_factor;
            if( is_symmetric ){ diag = fr_diag; };
            near_values = near->fr_values;
            far_values = far->fr_values;
            break;
        }
        case BCTKernelType::HighOrder:
        {
            far_factor = hi_far_factor;
            near_factor = hi_near_factor;
            if( is_symmetric ){ diag = hi_diag; };
            near_values = near->hi_values;
            far_values = far->hi_values;
            break;
        }
        case BCTKernelType::LowOrder:
        {
            far_factor = lo_far_factor;
            near_factor = lo_near_factor;
            if( is_symmetric ){ diag = lo_diag; };
            near_values = near->lo_values;
            far_values = far->lo_values;
            break;
        }
        default:
        {
            eprint("Unknown kernel. Doing nothing.");
            return;
        }
        }
        // The factor of 2. in the last argument stems from the symmetry of the kernel

        // TODO: In case of S != T, we have to replace each call with one call to ApplyKernel_CSR_MKL and one to (a yet to be written) ApplyKernelTranspose_CSR_MKL

        //    near->ApplyKernel_CSR_Eigen( near_values, T->P_in, S->P_out, cols, -2.0 * near_factor );
        //     far->ApplyKernel_CSR_Eigen(  far_values, T->C_in, S->C_out, cols, -2.0 * far_factor );

        near->ApplyKernel_CSR_MKL( near_values, T->P_in, S->P_out, cols, -2.0 * near_factor );
         far->ApplyKernel_CSR_MKL(  far_values, T->C_in, S->C_out, cols, -2.0 * far_factor );

        //     Adding product of diagonal matrix of "diags".
        if ( diag ){
            mint last = std::min( S->primitive_count, T->primitive_count );    // A crude safe-guard protecting against out-of-bound access if S != T.
            #pragma omp parallel for
            for( mint i = 0; i < last; ++i )
            {
                cblas_daxpy( cols, diag[i], T->P_in + (cols * i), 1, S->P_out + (cols * i), 1 );
            }
        }
        
        ptoc("OptimizedBlockClusterTree::InternalMultiply");
    }; // InternalMultiply

    // TODO: Needs to be adjusted when S and T are not the same!!!
    void OptimizedBlockClusterTree::ComputeDiagonals()
    {
        ptic("OptimizedBlockClusterTree::ComputeDiagonals");
        if( true )
        {
            S->RequireBuffers(1);
            T->RequireBuffers(1);

            //Sloppily: hi_diag = hi_ker * P_near[0], where hi_ker is the kernel implemented in ApplyKernel_CSR_MKL

            // Initialize the "diag" vector (weighted by the primitive weights)
            {
                mreal const  * restrict const a = T->P_near[0];
                mreal * restrict const diag = T->P_in;
                mint m = T->primitive_count;
                #pragma omp parallel for simd aligned( a, diag : ALIGN )
                for( mint i = 0; i < m; ++i )
                {
                    diag[i] = a[i];
                }
            }

    //        print("T->P_to_C.Multiply");
            T->P_to_C.Multiply( T->P_in, T->C_in, 1);
    //        tic("T->PercolateUp");
            T->PercolateUp();
    //        toc("T->PercolateUp");

            fr_diag = mreal_alloc( S->primitive_count );
            hi_diag = mreal_alloc( S->primitive_count );
            lo_diag = mreal_alloc( S->primitive_count );


            // The factor of 2. in the last argument stems from the symmetry of the kernel
            far->ApplyKernel_CSR_MKL(  far->fr_values, T->C_in, S->C_out, 1, 2. * fr_far_factor );
           near->ApplyKernel_CSR_MKL( near->fr_values, T->P_in, S->P_out, 1, 2. * fr_near_factor );

            S->PercolateDown();
            S->C_to_P.Multiply( S->C_out, S->P_out, 1, true);


            // TODO: Explain the hack of dividing by S->P_near[0][i] here to a future self so that he won't change this later.
            
            mreal * restrict ainv = mreal_alloc(S->primitive_count);
            mint m = S->primitive_count;
            mreal const * restrict const data = S->P_out;
            mreal const * restrict const a = S->P_near[0];
            
            #pragma omp parallel for simd aligned( ainv, a, fr_diag, data : ALIGN)
            for( mint i = 0; i < m; ++i )
            {
                ainv[i] = 1./(a[i]);
                fr_diag[i] =  ainv[i] * data[i];
            }


            far->ApplyKernel_CSR_MKL(  far->hi_values, T->C_in, S->C_out, 1, 2. * hi_far_factor );
           near->ApplyKernel_CSR_MKL( near->hi_values, T->P_in, S->P_out, 1, 2. * hi_near_factor );

            S->PercolateDown();
            S->C_to_P.Multiply( S->C_out, S->P_out, 1, true);

            #pragma omp parallel for simd aligned( ainv, hi_diag, data : ALIGN)
            for( mint i = 0; i < m; ++i )
            {
                hi_diag[i] =  ainv[i] * data[i];
            }

            far->ApplyKernel_CSR_MKL(  far->lo_values, T->C_in, S->C_out, 1, 2. * lo_far_factor );
           near->ApplyKernel_CSR_MKL( near->lo_values, T->P_in, S->P_out, 1, 2. * lo_near_factor );

            S->PercolateDown();
            S->C_to_P.Multiply( S->C_out, S->P_out, 1, true);

            #pragma omp parallel for simd aligned( ainv, lo_diag, data : ALIGN)
            for( mint i = 0; i < m; ++i )
            {
                lo_diag[i] =  ainv[i] * data[i];
            }
            
            mreal_free(ainv);
        }
        ptoc("OptimizedBlockClusterTree::ComputeDiagonals");
    }; // ComputeDiagonals


    void OptimizedBlockClusterTree::AddObstacleCorrection( OptimizedBlockClusterTree * bct12)
    {
        ptic("OptimizedBlockClusterTree::AddObstacleCorrection");
        // Suppose that bct11 = this;
        // The joint bct of the union of mesh1 and mesh2 can be written in block matrix for as
        //  bct = {
        //            { bct11, bct12 },
        //            { bct21, bct22 }
        //        },
        // where bct11 and bct22 are the instances of OptimizedBlockClusterTree of mesh1 and mesh2, respectively, bct12 is cross interaction OptimizedBlockClusterTree of mesh1 and mesh2, and bct21 is the transpose of bct12.
        // However, the according matrix (on the space of dofs on the primitives) would be
        //  A   = {
        //            { A11 + diag( A12 * one2 ) , A12                      },
        //            { A21                      , A22 + diag( A21 * one1 ) }
        //        },
        // where one1 and one2 are all-1-vectors on the primitives of mesh1 and mesh2, respectively.
        // OptimizedBlockClusterTree::AddObstacleCorrection is supposed to compute diag( A12 * one2 ) and to add it to the diagonal of A11.
        // Then the bct11->Multiply will also multiply with the obstacle.
        
        if( (S == T) && (T == bct12->S) )
        {
            RequireMetrics();
            bct12->RequireMetrics();

            if( fr_factor != bct12->fr_factor )
            {
                wprint("AddToDiagonal: The values of fr_factor of the two instances of OptimizedBlockClusterTree do not coincide.");
            }
            if( hi_factor != bct12->hi_factor )
            {
                wprint("AddToDiagonal: The values of hi_factor of the two instances of OptimizedBlockClusterTree do not coincide.");
            }
            if( lo_factor != bct12->lo_factor )
            {
                wprint("AddToDiagonal: The values of lo_factor of the two instances of OptimizedBlockClusterTree do not coincide.");
            }
            
            mint n = T->primitive_count;
            
            mreal * restrict const fr_target = fr_diag;
            mreal * restrict const hi_target = hi_diag;
            mreal * restrict const lo_target = lo_diag;
            
            mreal const * restrict const fr_source = bct12->fr_diag;
            mreal const * restrict const hi_source = bct12->hi_diag;
            mreal const * restrict const lo_source = bct12->lo_diag;
            
            #pragma omp parallel for simd aligned( fr_target, hi_target, lo_target, fr_source, hi_source, lo_source : ALIGN )
            for( mint i = 0; i < n; ++ i)
            {
                fr_target[i] += fr_source[i];
                hi_target[i] += hi_source[i];
                lo_target[i] += lo_source[i];
            }
        }
        else
        {
            if( S != T )
            {
                eprint("AddToDiagonal: Instance of OptimizedBlockClusterTree is not symmetric. Doing nothing.");
            }
            if( S != bct12->S )
            {
                eprint("AddToDiagonal: The two instances of OptimizedBlockClusterTree are not compatible. Doing nothing.");
            }
        }
        ptoc("OptimizedBlockClusterTree::AddObstacleCorrection");
    }
} // namespace rsurfaces
