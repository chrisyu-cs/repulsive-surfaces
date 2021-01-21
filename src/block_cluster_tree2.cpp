#include "block_cluster_tree2.h"

namespace rsurfaces
{

    BlockClusterTree2::BlockClusterTree2(ClusterTree2 *S_, ClusterTree2 *T_, const mreal alpha_, const mreal beta_, const mreal theta_, bool exploit_symmetry_, bool upper_triangular_)
    {
        // tic("Initializing BlockClusterTree2");
        S = S_;
        T = T_;
        alpha = alpha_;
        beta = beta_;
        theta2 = theta_ * theta_;
        exploit_symmetry = exploit_symmetry_;
        upper_triangular = upper_triangular_;
        disableNearField = false;

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
        }

        // print("thread_count = " + std::to_string(thread_count) );

        // tic("CreateBlockClusters");
        CreateBlockClusters();
        // toc("CreateBlockClusters");

        //    mint err;
        //    tic("Check far");
        //        err = far->Check();
        //    toc("Check");
        //
        //    tic("Check near");
        //        err = near->Check();
        //    toc("Check");

        // tic("Test Energy");
        // tic("FarFieldEnergy");
        // mreal en_far = FarFieldEnergy();
        // print("FarFieldEnergy (multipole) = " + std::to_string(en_far) );
        // toc("FarFieldEnergy");
        // tic("NearFieldEnergy");
        // mreal en_near = NearFieldEnergy();
        // print("NearFieldEnergy (multipole) = " + std::to_string(en_near) );
        // toc("NearFieldEnergy");

        // print("total energy (multipole) = " + std::to_string( en_far + en_near ) );

        // toc("Test Energy");

        // tic("Interactions");

        // tic("FarFieldInteraction");
        FarFieldInteraction();
        // toc("FarFieldInteraction");

        // tic("NearFieldInteractionCSR");
        NearFieldInteractionCSR();
        // toc("NearFieldInteractionCSR");

        // toc("Interactions");

        // toc("Initializing BlockClusterTree2");

        // tic("ComputeDiagonals");
        ComputeDiagonals();
        // toc("ComputeDiagonals");

        //    tic("Test Multiply");
        //
        //    mint cols = 9;
        //    A_Vector<mreal> V (T->primitive_count * cols, 1.);
        //    A_Vector<mreal> U (S->primitive_count * cols);
        //    Multiply( &V[0], &U[0], cols, BCTKernelType::HighOrder );
        //
        //
        //    toc("Test Multiply");

        // print("BlockClusterTree2 initialized.");
    }; // Constructor

    //######################################################################################################################################
    //      Initialization
    //######################################################################################################################################

    void BlockClusterTree2::CreateBlockClusters()
    {
        auto thread_sep_idx = A_Vector<A_Deque<mint>>(thread_count);
        auto thread_sep_jdx = A_Vector<A_Deque<mint>>(thread_count);

        auto thread_nonsep_idx = A_Vector<A_Deque<mint>>(thread_count);
        auto thread_nonsep_jdx = A_Vector<A_Deque<mint>>(thread_count);

        // tic("SplitBlockCluster");

#pragma omp parallel num_threads(thread_count) default(none) shared(thread_sep_idx, thread_sep_jdx, thread_nonsep_idx, thread_nonsep_jdx)
        {
#pragma omp single
            {
                SplitBlockCluster(thread_sep_idx, thread_sep_jdx, thread_nonsep_idx, thread_nonsep_jdx, 0, 0, thread_count);
            }
        }

        mint sep_blockcluster_count = 0;
        mint nonsep_blockcluster_count = 0;

        for (mint thread = 0; thread < thread_count; ++thread)
        {
            sep_blockcluster_count += thread_sep_idx[thread].size();
            nonsep_blockcluster_count += thread_nonsep_idx[thread].size();
        }

        // print("sep_blockcluster_count = "+std::to_string(sep_blockcluster_count));

        // print("nonsep_blockcluster_count = "+std::to_string(nonsep_blockcluster_count));

        // toc("SplitBlockCluster ");

        // tic("Creation of sparsity patterns");

        // using parallel count sort to sort the cluster (i,j)-pairs according to i.

        // tic("Far field");

        far = std::make_shared<InteractionData>(thread_sep_idx, thread_sep_jdx,
                                                S->cluster_count, T->cluster_count, upper_triangular);

        // deallocation to free space for the nonsep_row_pointers
        thread_sep_idx = A_Vector<A_Deque<mint>>();
        thread_sep_jdx = A_Vector<A_Deque<mint>>();

        // toc("Far field");

        // tic("Near field");

        near = std::make_shared<InteractionData>(thread_nonsep_idx, thread_nonsep_jdx,
                                                 S->primitive_count, T->primitive_count, S->leaf_cluster_ptr, T->leaf_cluster_ptr, upper_triangular);

        // toc("Near field");

        // toc(" Creation of sparsity patterns");

    }; //CreateBlockClusters

    void BlockClusterTree2::SplitBlockCluster(
        A_Vector<A_Deque<mint>> &sep_i,
        A_Vector<A_Deque<mint>> &sep_j,
        A_Vector<A_Deque<mint>> &nsep_i,
        A_Vector<A_Deque<mint>> &nsep_j,
        const mint i,
        const mint j,
        const mint free_thread_count)
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
#pragma omp task final(free_thread_count < 1) default(none) firstprivate(lefti, leftj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, lefti, leftj, spawncount + (remainder > 0));
#pragma omp task final(free_thread_count < 1) default(none) firstprivate(lefti, rightj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, lefti, rightj, spawncount + (remainder > 2));
#pragma omp task final(free_thread_count < 1) default(none) firstprivate(righti, rightj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, righti, rightj, spawncount);
                        //                    #pragma omp taskwait
                    }
                    else
                    {
                        // In case of exploit_symmetry !=0, this is a very seldom case; still requird to preserve symmetry.
                        // This happens only if i and j represent _diffent clusters with same radii.

                        mint spawncount = free_thread_count / 4;
                        mint remainder = free_thread_count % 4;

#pragma omp task final(free_thread_count < 1) default(none) firstprivate(lefti, leftj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, lefti, leftj, spawncount + (remainder > 0));
#pragma omp task final(free_thread_count < 1) default(none) firstprivate(righti, leftj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, righti, leftj, spawncount + (remainder > 1));
#pragma omp task final(free_thread_count < 1) default(none) firstprivate(lefti, rightj, spawncount, remainder) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, lefti, rightj, spawncount + (remainder > 2));
#pragma omp task final(free_thread_count < 1) default(none) firstprivate(righti, rightj, spawncount) shared(sep_i, sep_j, nsep_i, nsep_j)
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
#pragma omp task final(free_thread_count < 1) default(none) firstprivate(lefti, j, free_thread_count) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, lefti, j, free_thread_count / 2);
#pragma omp task final(free_thread_count < 1) default(none) firstprivate(righti, j, free_thread_count) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, righti, j, free_thread_count - free_thread_count / 2);
                        //                    #pragma omp taskwait
                    }
                    else //scorei < scorej
                    {
//split cluster j
#pragma omp task final(free_thread_count < 1) default(none) firstprivate(i, leftj, free_thread_count) shared(sep_i, sep_j, nsep_i, nsep_j)
                        SplitBlockCluster(sep_i, sep_j, nsep_i, nsep_j, i, leftj, free_thread_count / 2);
#pragma omp task final(free_thread_count < 1) default(none) firstprivate(i, rightj, free_thread_count) shared(sep_i, sep_j, nsep_i, nsep_j)
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

    // Coarse energy approximation to test whether BlockClusterTree2 is working correctly.
    mreal BlockClusterTree2::FarFieldEnergy()
    {
        mint b_m = far->b_m;
        mint const *const restrict b_outer = &far->b_outer[0];
        mint const *const restrict b_inner = &far->b_inner[0];

        // Dunno why "restrict" helps with P_data. It is actually a lie here when S = T.
        mreal const *const restrict A = &S->C_data[0][0];
        mreal const *const restrict X1 = &S->C_data[1][0];
        mreal const *const restrict N1 = &S->C_data[4][0];
        mreal const *const restrict X2 = &S->C_data[2][0];
        mreal const *const restrict N2 = &S->C_data[5][0];
        mreal const *const restrict X3 = &S->C_data[3][0];
        mreal const *const restrict N3 = &S->C_data[6][0];

        mreal const *const restrict B = &T->C_data[0][0];
        mreal const *const restrict Y1 = &T->C_data[1][0];
        mreal const *const restrict M1 = &T->C_data[4][0];
        mreal const *const restrict Y2 = &T->C_data[2][0];
        mreal const *const restrict M2 = &T->C_data[5][0];
        mreal const *const restrict Y3 = &T->C_data[3][0];
        mreal const *const restrict M3 = &T->C_data[6][0];

        mreal sum = 0.;
#pragma omp parallel for num_threads(thread_count) schedule(guided, 8) reduction(+ \
                                                                                 : sum)
        for (mint i = 0; i < b_m; ++i)
        {
            mreal x1 = X1[i];
            mreal n1 = N1[i];
            mreal x2 = X2[i];
            mreal n2 = N2[i];
            mreal x3 = X3[i];
            mreal n3 = N3[i];

            mreal local_sum = 0.;

            // This loop can be SIMDized straight-forwardly (horizontal SIMDization).
            for (mint k = b_outer[i]; k < b_outer[i + 1]; ++k)
            {
                mint j = b_inner[k];

                if (i <= j)
                {
                    mreal v1 = Y1[j] - x1;
                    mreal m1 = M1[j];
                    mreal v2 = Y2[j] - x2;
                    mreal m2 = M2[j];
                    mreal v3 = Y3[j] - x3;
                    mreal m3 = M3[j];

                    mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                    mreal rCosPsi = v1 * m1 + v2 * m2 + v3 * m3;
                    mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;

                    //                    mreal en = ( pow( fabs(rCosPhi(, alpha ) + pow( fabs(rCosPsi), alpha) ) * pow( r2, -0.5 * beta );

                    // Avoiding pow and sqrt for performance; hard coded to for alpha = 6 and beta = 12
                    mreal rinv2 = 1. / r2;
                    mreal rinv6 = rinv2 * rinv2 * rinv2;
                    mreal rinv12 = rinv6 * rinv6;
                    mreal rCosPhi2 = rCosPhi * rCosPhi;
                    mreal rCosPsi2 = rCosPsi * rCosPsi;
                    mreal en = (rCosPhi2 * rCosPhi2 * rCosPhi2 + rCosPsi2 * rCosPsi2 * rCosPsi2) * rinv12;
                    local_sum += en * B[j];
                }
            }

            sum += A[i] * local_sum;
        }
        return sum;
    }; //FarFieldEnergy

    mreal BlockClusterTree2::NearFieldEnergy()
    {
        // Caution: This functions assumes that S = T!!!

        // Coarse energy approximation to test whether BlockClusterTree2 is working correctly.

        mint b_m = near->b_m;

        mint const *const restrict b_row_ptr = &near->b_row_ptr[0];
        mint const *const restrict b_col_ptr = &near->b_col_ptr[0];
        mint const *const restrict b_outer = &near->b_outer[0];
        mint const *const restrict b_inner = &near->b_inner[0];

        // Dunno why "restrict" helps with P_data. It is actually a lie here.
        mreal const *const restrict A = &S->P_data[0][0];
        mreal const *const restrict X1 = &S->P_data[1][0];
        mreal const *const restrict N1 = &S->P_data[4][0];
        mreal const *const restrict X2 = &S->P_data[2][0];
        mreal const *const restrict N2 = &S->P_data[5][0];
        mreal const *const restrict X3 = &S->P_data[3][0];
        mreal const *const restrict N3 = &S->P_data[6][0];

        mreal const *const restrict B = &T->P_data[0][0];
        mreal const *const restrict Y1 = &T->P_data[1][0];
        mreal const *const restrict M1 = &T->P_data[4][0];
        mreal const *const restrict Y2 = &T->P_data[2][0];
        mreal const *const restrict M2 = &T->P_data[5][0];
        mreal const *const restrict Y3 = &T->P_data[3][0];
        mreal const *const restrict M3 = &T->P_data[6][0];

        mreal sum = 0.;
#pragma omp parallel for num_threads(thread_count) schedule(guided, 8) reduction(+ \
                                                                                 : sum)
        for (mint b_i = 0; b_i < b_m; ++b_i)
        {

            mint i_begin = b_row_ptr[b_i];
            mint i_end = b_row_ptr[b_i + 1];

            for (mint k = b_outer[b_i]; k < b_outer[b_i + 1]; ++k)
            {
                mint b_j = b_inner[k];
                if (b_i <= b_j)
                {
                    mint j_begin = b_col_ptr[b_j];
                    mint j_end = b_col_ptr[b_j + 1];
                    mreal block_sum = 0.;

                    for (mint i = i_begin; i < i_end; ++i)
                    {
                        mreal x1 = X1[i];
                        mreal n1 = N1[i];
                        mreal x2 = X2[i];
                        mreal n2 = N2[i];
                        mreal x3 = X3[i];
                        mreal n3 = N3[i];

                        mreal i_sum = 0.;

                        // Here, one could do a bit of horizontal vectorization. However, the number of js an x interacts with varies greatly..
                        for (mint j = (b_i != b_j ? j_begin : i + 1); j < j_end; ++j) // if i == j, we loop only over the upper triangular block, diagonal excluded
                        {
                            mreal v1 = Y1[j] - x1;
                            mreal m1 = M1[j];
                            mreal v2 = Y2[j] - x2;
                            mreal m2 = M2[j];
                            mreal v3 = Y3[j] - x3;
                            mreal m3 = M3[j];

                            mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                            mreal rCosPsi = v1 * m1 + v2 * m2 + v3 * m3;
                            mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;

                            //                        mreal en = ( pow( fabs(rCosPhi), alpha ) + pow( fabs(rCosPsi), alpha) ) * pow( r2, -0.5 * beta );

                            // Avoiding pow and sqrt for performance; hard coded to for alpha = 6 and beta = 12
                            mreal rinv2 = 1. / r2;
                            mreal rinv6 = rinv2 * rinv2 * rinv2;
                            mreal rinv12 = rinv6 * rinv6;
                            mreal rCosPhi2 = rCosPhi * rCosPhi;
                            mreal rCosPsi2 = rCosPsi * rCosPsi;
                            mreal en = (rCosPhi2 * rCosPhi2 * rCosPhi2 + rCosPsi2 * rCosPsi2 * rCosPsi2) * rinv12;

                            i_sum += en * B[j];
                        }
                        block_sum += A[i] * i_sum;
                    }
                    //            print( "{ " + std::to_string(i) + " , " + std::to_string(j) + "} -> " + std::to_string(block_sum) );
                    sum += block_sum;
                }
            }
        }
        return sum;
    }; //NearFieldEnergy

    void BlockClusterTree2::FarFieldInteraction()
    {
        mint b_m = far->b_m;
        mint const *const restrict b_outer = &far->b_outer[0];
        mint const *const restrict b_inner = &far->b_inner[0];

        // Dunno why "restrict" helps with C_data. It is actually a lie here.
        mreal const *const restrict X1 = &S->C_data[1][0];
        mreal const *const restrict N1 = &S->C_data[4][0];
        mreal const *const restrict X2 = &S->C_data[2][0];
        mreal const *const restrict N2 = &S->C_data[5][0];
        mreal const *const restrict X3 = &S->C_data[3][0];
        mreal const *const restrict N3 = &S->C_data[6][0];

        mreal const *const restrict Y1 = &T->C_data[1][0];
        mreal const *const restrict M1 = &T->C_data[4][0];
        mreal const *const restrict Y2 = &T->C_data[2][0];
        mreal const *const restrict M2 = &T->C_data[5][0];
        mreal const *const restrict Y3 = &T->C_data[3][0];
        mreal const *const restrict M3 = &T->C_data[6][0];

        // "restrict" makes sense to me here because it exclude write conflicts.
        mreal *const restrict fr_values = &far->fr_values[0];
        mreal *const restrict hi_values = &far->hi_values[0];
        mreal *const restrict lo_values = &far->lo_values[0];

// Using i and j for cluster positions.
#pragma omp parallel for num_threads(thread_count) schedule(guided, 8)
        for (mint i = 0; i < b_m; ++i)
        {
            mreal x1 = X1[i];
            mreal n1 = N1[i];
            mreal x2 = X2[i];
            mreal n2 = N2[i];
            mreal x3 = X3[i];
            mreal n3 = N3[i];

            // This loop can be SIMDized straight-forwardly (horizontal SIMDization).
            // It is in no way the bottleneck at the moment. BlockClusterTree2::NearFieldEnergy takes many times longer.
            for (mint k = b_outer[i]; k < b_outer[i + 1]; ++k)
            {
                mint j = b_inner[k]; // We are in  block {i, j}

                mreal v1 = Y1[j] - x1;
                mreal m1 = M1[j];
                mreal v2 = Y2[j] - x2;
                mreal m2 = M2[j];
                mreal v3 = Y3[j] - x3;
                mreal m3 = M3[j];

                mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                mreal rCosPsi = v1 * m1 + v2 * m2 + v3 * m3;
                mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;

                mreal hi = mypow(r2, hi_exponent); // I got it down to this single call to pow. We might want to generate a lookup table for it...

                hi_values[k] = 2.0 * hi; // The factor 2.0 might be suboptimal. That's what my Mathematica code uses and it seems to work fine.

                //                        near->fr_values[ptr] = pow( r2, fr_exponent );
                mreal mul = r2;
                for (mint l = 0; l < intrinsic_dim; ++l)
                {
                    mul *= r2;
                }
                // mul = pow( r2, 1 + dim);
                fr_values[k] = 1. / (hi * mul);

                lo_values[k] = (rCosPhi * rCosPhi + rCosPsi * rCosPsi) / (r2 * r2) * hi;
            }
        }
    }; //FarFieldInteraction

    void BlockClusterTree2::NearFieldInteractionCSR()
    {
        mint b_m = near->b_m;
        // Getting the pointers first to reduce indexing within the loops. Together with const + restrict, this gains about 5% runtime improvement.
        mint const *const restrict b_row_ptr = &near->b_row_ptr[0];
        mint const *const restrict b_col_ptr = &near->b_col_ptr[0];
        mint const *const restrict b_outer = &near->b_outer[0];
        mint const *const restrict b_inner = &near->b_inner[0];
        mint const *const restrict outer = &near->outer[0];

        // Dunno why "restrict" helps with P_data. It is actually a lie here if S = S.
        mreal const *const restrict X1 = &S->P_data[1][0];
        mreal const *const restrict N1 = &S->P_data[4][0];
        mreal const *const restrict X2 = &S->P_data[2][0];
        mreal const *const restrict N2 = &S->P_data[5][0];
        mreal const *const restrict X3 = &S->P_data[3][0];
        mreal const *const restrict N3 = &S->P_data[6][0];

        mreal const *const restrict Y1 = &T->P_data[1][0];
        mreal const *const restrict M1 = &T->P_data[4][0];
        mreal const *const restrict Y2 = &T->P_data[2][0];
        mreal const *const restrict M2 = &T->P_data[5][0];
        mreal const *const restrict Y3 = &T->P_data[3][0];
        mreal const *const restrict M3 = &T->P_data[6][0];

        // "restrict" makes sense to me here because it exclude write conflicts.
        mreal *const restrict fr_values = &near->fr_values[0];
        mreal *const restrict hi_values = &near->hi_values[0];
        mreal *const restrict lo_values = &near->lo_values[0];

// Using b_i and b_j for block (leaf cluster) positions.
// Using i and j for primitive positions.
#pragma omp parallel for
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
                mreal n1 = N1[i];
                mreal x2 = X2[i];
                mreal n2 = N2[i];
                mreal x3 = X3[i];
                mreal n3 = N3[i];

                // From here on, the read-access to T->P_data is a bit cache-unfriendly.
                // However, the threads write to disjoint large consecutive blocks of memory. Because write is consecutively, false sharing is probably not an issue.

                for (mint k = k_begin; k < k_end; ++k) // loop over all blocks in block row b_i
                {

                    mint b_j = b_inner[k]; // we are in block {b_i, b_j} now

                    mint j_begin = b_col_ptr[b_j];
                    mint j_end = b_col_ptr[b_j + 1];

                    for (mint j = j_begin; j < j_end; ++j)
                    {
                        if (i != j)
                        {
                            mreal v1 = Y1[j] - x1;
                            mreal m1 = M1[j];
                            mreal v2 = Y2[j] - x2;
                            mreal m2 = M2[j];
                            mreal v3 = Y3[j] - x3;
                            mreal m3 = M3[j];

                            mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                            mreal rCosPsi = v1 * m1 + v2 * m2 + v3 * m3;
                            mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;

                            // The following line makes up approx 2/3 of this function's runtime! This is whe we avoid pow as much as possible and replace it with mypow.
                            mreal hi = mypow(r2, hi_exponent); // I got it down to this single call to pow. We might want to generate a lookup table for it...

                            hi_values[ptr] = 2.0 * hi; // The factor 2.0 might be suboptimal. That's what my Mathematica code uses (somwwhat accidentally) and it seems to work fine.

                            //                        near->fr_values[ptr] = pow( r2, fr_exponent );
                            mreal mul = r2;
                            for (mint l = 0; l < intrinsic_dim; ++l)
                            {
                                mul *= r2;
                            }
                            // mul = pow( r2, 1 + dim);
                            fr_values[ptr] = 1. / (hi * mul);

                            lo_values[ptr] = (rCosPhi * rCosPhi + rCosPsi * rCosPsi) / (r2 * r2) * hi;
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
    }; //NearFieldInteractionCSR

    //######################################################################################################################################
    //      Vector multiplication
    //######################################################################################################################################

    void BlockClusterTree2::Multiply(Eigen::VectorXd &input, Eigen::VectorXd &output, BCTKernelType type, bool addToResult) const
    {
        Multiply(input, output, 1, type, addToResult);
    }

    void BlockClusterTree2::Multiply(Eigen::VectorXd &input, Eigen::VectorXd &output, const mint k, BCTKernelType type, bool addToResult) const
    {
        // Version for vectors of k-dimensional vectors. Input and out are assumed to be stored in interleave format.
        // E.g., for a list {v1, v2, v3,...} of  k = 3-vectors, we expect { v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z, ... }

        mint n = T->lo_pre.n; // Expected length for a vector of scalars

        if ((input.size() >= k * n) && (output.size() >= k * n))
        {
            // the acual multiplocation
            T->Pre(input.data(), k, type);
            InternalMultiply(type);
            S->Post(output.data(), k, type, addToResult);

            if (type == BCTKernelType::HighOrder || type == BCTKernelType::LowOrder)
            {
                output /= 2;
            }

            //        // Just copy to /add into the remaining entries.
            //        mint nRemainingEntries = std::min(input.size() , output.size() ) - k * n;
            //        if(addToResult)
            //        {
            //            output.segment(k * n, nRemainingEntries) += input.segment(k * n, nRemainingEntries);
            //        }
            //        else
            //        {
            //            output.segment(k * n, nRemainingEntries)  = input.segment(k * n, nRemainingEntries);
            //        }
        }
        else
        {
            if ((input.size() < k * n))
            {
                eprint(" in BlockClusterTree2::Multiply: input vector is to short because" + std::to_string(input.size()) + " < " + std::to_string(k) + " * " + std::to_string(n) + ".");
            }
            if ((output.size() < k * n))
            {
                eprint(" in BlockClusterTree2::Multiply: input vector is to short because" + std::to_string(output.size()) + " < " + std::to_string(k) + " * " + std::to_string(n) + ".");
            }
        }
    }

    //######################################################################################################################################
    //      Matrix multiplication
    //######################################################################################################################################

    void BlockClusterTree2::Multiply(Eigen::MatrixXd &input, Eigen::MatrixXd &output, BCTKernelType type, bool addToResult) const
    {
        // Top level routine for the user.
        // Optimized for in and output in row major order.

        //    tic("Multiply");

        //    tic("T->Pre");
        T->Pre(input, type);
        //    toc("T->Pre");

        //    tic("InternalMultiply");
        InternalMultiply(type);
        //    toc("InternalMultiply");

        //    tic("S->Post");
        S->Post(output, type, addToResult);
        //    toc("S->Post");

        //    toc("Multiply");
    }; // Multiply

    void BlockClusterTree2::InternalMultiply(BCTKernelType type) const
    {

        mreal *diag = NULL;
        A_Vector<mreal> *near_values;
        A_Vector<mreal> *far_values;

        mreal factor = 1.;

        mint cols = T->buffer_dim;

        S->PrepareBuffers(cols); // Tell the S-side what it has to expect.

        // I know, this looks awful... hash tables with keys from BCTKernelType would be nicer.
        switch (type)
        {
        case BCTKernelType::FractionalOnly:
        {
            factor = fr_factor;
            diag = &fr_diag[0];
            near_values = &near->fr_values;
            far_values = &far->fr_values;
            break;
        }
        case BCTKernelType::HighOrder:
        {
            factor = hi_factor;
            diag = &hi_diag[0];
            near_values = &near->hi_values;
            far_values = &far->hi_values;
            break;
        }
        case BCTKernelType::LowOrder:
        {
            factor = lo_factor;
            diag = &lo_diag[0];
            near_values = &near->lo_values;
            far_values = &far->lo_values;
            break;
        }
        default:
        {
            eprint("Unknown kernel. Doing nothing.");
            return;
        }
        }

        //    tic("Apply");
        // The factor of 2. in the last argument stems from the symmetry of the kernel

        // TODO: In case of S != T, we have to replace each call to by one call to ApplyKernel_CSR_MKL and one to (a yet to be written) ApplyKernelTranspose_CSR_MKL

        //    near->ApplyKernel_CSR_Eigen( *near_values, &T->P_in[0], &S->P_out[0], cols, -2.0 * factor );
        //     far->ApplyKernel_CSR_Eigen(  *far_values, &T->C_in[0], &S->C_out[0], cols, -2.0 * factor );

        // if (!disableNearField)
        // {
            //    tic("near MKL");
            near->ApplyKernel_CSR_MKL(*near_values, &T->P_in[0], &S->P_out[0], cols, -2.0 * factor);
            //    toc("near MKL");
        // }

        //    tic("far MKL");
        far->ApplyKernel_CSR_MKL(*far_values, &T->C_in[0], &S->C_out[0], cols, -2.0 * factor);
        //    toc("far MKL");
        //    toc("Apply");

        //     Adding product of diagonal matrix of "diags".
        // TODO: We want to skip this if S != T.
        if (diag)
        {
            //        tic("Diagonal");
            //        #pragma omp parallel for
            for (mint i = 0; i < std::min(S->primitive_count, T->primitive_count); ++i) // A crude safe-guard protecting against out-of-bound access if S != T.
            {
                cblas_daxpy(cols, diag[i], &T->P_in[cols * i], 1, &S->P_out[cols * i], 1);

                //             Compiler will likely be able to vectorize this
                //            mreal val = diag[i];
                //            for( mint k = cols * i; k < cols * (i+1) ; ++k )
                //            {
                //                S->P_out[k] += val * T->P_in[k];
                //            }
            }
            //        toc("Diagonal");
        }

    }; // InternalMultiply

    // TODO: Needs to be adjusted when S and T are not the same!!!
    void BlockClusterTree2::ComputeDiagonals()
    {
        //    A_Vector<mreal> diag ( S->primitive_count, 1. );
        S->PrepareBuffers(1);
        T->PrepareBuffers(1);

        //Sloppily: hi_diag = hi_ker * P_data[0], where hi_ker is the kernel implemented in ApplyKernel_CSR_MKL

        // Initialize the "diag" vector (weighted by the primitive weights
        std::copy(T->P_data[0].begin(), T->P_data[0].begin() + T->primitive_count, T->P_in.begin());

        T->P_to_C.Multiply(&T->P_in[0], &T->C_in[0], 1);

        T->PercolateUp(0, thread_count);

        fr_diag = A_Vector<mreal>(S->primitive_count);
        hi_diag = A_Vector<mreal>(S->primitive_count);
        lo_diag = A_Vector<mreal>(S->primitive_count);

        // The factor of 2. in the last argument stems from the symmetry of the kernel
        far->ApplyKernel_CSR_MKL(far->fr_values, &T->C_in[0], &S->C_out[0], 1, 2. * fr_factor);
        near->ApplyKernel_CSR_MKL(near->fr_values, &T->P_in[0], &S->P_out[0], 1, 2. * fr_factor);
        S->PercolateDown(0, thread_count);
        S->C_to_P.Multiply(&S->C_out[0], &S->P_out[0], 1, true);

        // TODO: Explain the hack of dividing by S->P_data[0][i] here to a future self so that he won't change this later.
        A_Vector<mreal> ainv(S->primitive_count);

#pragma omp parallel for
        for (mint i = 0; i < S->primitive_count; ++i)
        {
            ainv[i] = 1. / (S->P_data[0][i]);
            fr_diag[i] = ainv[i] * (S->P_out[i]);
        }

        far->ApplyKernel_CSR_MKL(far->hi_values, &T->C_in[0], &S->C_out[0], 1, 2. * hi_factor);
        near->ApplyKernel_CSR_MKL(near->hi_values, &T->P_in[0], &S->P_out[0], 1, 2. * hi_factor);

        S->PercolateDown(0, thread_count);
        S->C_to_P.Multiply(&S->C_out[0], &S->P_out[0], 1, true);

#pragma omp parallel for
        for (mint i = 0; i < S->primitive_count; ++i)
        {
            hi_diag[i] = ainv[i] * (S->P_out[i]);
        }

        far->ApplyKernel_CSR_MKL(far->lo_values, &T->C_in[0], &S->C_out[0], 1, 2. * lo_factor);
        near->ApplyKernel_CSR_MKL(near->lo_values, &T->P_in[0], &S->P_out[0], 1, 2. * lo_factor);
        S->PercolateDown(0, thread_count);
        S->C_to_P.Multiply(&S->C_out[0], &S->P_out[0], 1, true);
#pragma omp parallel for
        for (mint i = 0; i < S->primitive_count; ++i)
        {
            lo_diag[i] = ainv[i] * (S->P_out[i]);
        }

    }; // ComputeDiagonals

} // namespace rsurfaces