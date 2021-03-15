
#include "energy/tp_obstacle_barnes_hut_pr_0.h"

namespace rsurfaces
{
    double TPObstacleBarnesHut_Projectors0::Value()
    {
        bvh = bvhSharedFrom->GetBVH();
        if (!bvh)
        {
            throw std::runtime_error("Obstacle energy is sharing BVH from an energy that has no BVH.");
        }
        if (use_int)
        {
            mint int_alphahalf = std::round(alpha/2);
            mint int_betahalf = std::round(beta / 2);
            return weight * Energy(int_alphahalf, int_betahalf);
        }
        else
        {
            mreal real_alphahalf = alpha/2;
            mreal real_betahalf = beta / 2;
            return weight * Energy(real_alphahalf, real_betahalf);
        }
    } // Value

    void TPObstacleBarnesHut_Projectors0::Differential(Eigen::MatrixXd &output)
    {
        bvh = bvhSharedFrom->GetBVH();
        if (!bvh)
        {
            throw std::runtime_error("Obstacle energy is sharing BVH from an energy that has no BVH.");
        }
        
        if( bvh->data_dim != 10)
        {
            eprint("in TPObstacleBarnesHut_Projectors0::Differential: data_dim != 10");
        }
        
        EigenMatrixRM P_D_data ( bvh->primitive_count, 10 );
        
        bvh->CleanseD();
        
        if( use_int )
        {
            mint int_alpha = std::round(alpha);
            mint int_betahalf = std::round(beta/2);
            DEnergy( int_alpha, int_betahalf );
            
        }
        else
        {
            mreal real_alpha = alpha;
            mreal real_betahalf = beta/2;
            DEnergy( real_alpha, real_betahalf );
        }
        
        bvh->CollectDerivatives( P_D_data.data() );
    
        AssembleDerivativeFromACNData( mesh, geom, P_D_data, output, weight );
  
    } // Differential
    
    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void TPObstacleBarnesHut_Projectors0::Update()
    {
        // Invalidate the old BVH pointer
        bvh = 0;
        // bvhSharedFrom is responsible for reallocating it in its Update() function
        bvh = bvhSharedFrom->GetBVH();
        if (!bvh)
        {
            throw std::runtime_error("Obstacle energy is sharing BVH from an energy that has no BVH.");
        }
    }

    // Get the mesh associated with this energy.
    MeshPtr TPObstacleBarnesHut_Projectors0::GetMesh()
    {
        return mesh;
    }

    // Get the geometry associated with this geometry.
    GeomPtr TPObstacleBarnesHut_Projectors0::GetGeom()
    {
        return geom;
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 TPObstacleBarnesHut_Projectors0::GetExponents()
    {
        return Vector2{alpha, beta};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *TPObstacleBarnesHut_Projectors0::GetBVH()
    {
        return o_bvh;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double TPObstacleBarnesHut_Projectors0::GetTheta()
    {
        return theta;
    }

    template <typename T1, typename T2>
    mreal TPObstacleBarnesHut_Projectors0::Energy(T1 alpha, T2 betahalf)
    {
        T2 minus_betahalf = -betahalf;
        mreal theta2 = theta * theta;

        mint nthreads = bvh->thread_count;

        mreal sum = 0.;

        {
            auto S = bvh;
            auto T = o_bvh;
            mreal const *const restrict C_xmin1 = S->C_min[0];
            mreal const *const restrict C_xmin2 = S->C_min[1];
            mreal const *const restrict C_xmin3 = S->C_min[2];
            mreal const *const restrict C_xmax1 = S->C_max[0];
            mreal const *const restrict C_xmax2 = S->C_max[1];
            mreal const *const restrict C_xmax3 = S->C_max[2];

            mreal const *const restrict C_xr2 = S->C_squared_radius;

            mreal const *const restrict P_A = S->P_data[0];
            mreal const *const restrict P_X1 = S->P_data[1];
            mreal const *const restrict P_X2 = S->P_data[2];
            mreal const *const restrict P_X3 = S->P_data[3];
            mreal const *const restrict P_N1 = S->P_data[4];
            mreal const *const restrict P_N2 = S->P_data[5];
            mreal const *const restrict P_N3 = S->P_data[6];

            mint const *const restrict C_xbegin = S->C_begin;
            mint const *const restrict C_xend = S->C_end;

            mint const *const restrict leaf = S->leaf_clusters;

            mreal const *const restrict C_ymin1 = T->C_min[0];
            mreal const *const restrict C_ymin2 = T->C_min[1];
            mreal const *const restrict C_ymin3 = T->C_min[2];
            mreal const *const restrict C_ymax1 = T->C_max[0];
            mreal const *const restrict C_ymax2 = T->C_max[1];
            mreal const *const restrict C_ymax3 = T->C_max[2];

            mreal const *const restrict C_yr2 = T->C_squared_radius;

            mreal const *const restrict P_B = T->P_data[0];
            mreal const *const restrict P_Y1 = T->P_data[1];
            mreal const *const restrict P_Y2 = T->P_data[2];
            mreal const *const restrict P_Y3 = T->P_data[3];

            mreal const *const restrict C_B = T->C_data[0];
            mreal const *const restrict C_Y1 = T->C_data[1];
            mreal const *const restrict C_Y2 = T->C_data[2];
            mreal const *const restrict C_Y3 = T->C_data[3];

            mint const *const restrict C_ybegin = T->C_begin;
            mint const *const restrict C_yend = T->C_end;

            mint const *const restrict C_left = T->C_left;
            mint const *const restrict C_right = T->C_right;

            A_Vector<A_Vector<mint>> thread_stack(nthreads);

#pragma omp parallel for num_threads(nthreads) reduction(+ \
                                                         : sum)
            for (mint k = 0; k < S->leaf_cluster_count; ++k)
            {
                mint thread = omp_get_thread_num();

                A_Vector<mint> *stack = &thread_stack[thread];

                stack->clear();
                stack->push_back(0);

                mint l = leaf[k];
                mint i_begin = C_xbegin[l];
                mint i_end = C_xend[l];

                mreal xmin1 = C_xmin1[l];
                mreal xmin2 = C_xmin2[l];
                mreal xmin3 = C_xmin3[l];

                mreal xmax1 = C_xmax1[l];
                mreal xmax2 = C_xmax2[l];
                mreal xmax3 = C_xmax3[l];

                mreal r2l = C_xr2[l];

                mreal local_sum = 0.;

                while (!stack->empty())
                {
                    mint C = stack->back();
                    stack->pop_back();

                    mreal h2 = std::max(r2l, C_yr2[C]);

                    // Compute squared distance between bounding boxes.
                    // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares

                    mreal ymin1 = C_ymin1[C];
                    mreal ymin2 = C_ymin2[C];
                    mreal ymin3 = C_ymin3[C];

                    mreal ymax1 = C_ymax1[C];
                    mreal ymax2 = C_ymax2[C];
                    mreal ymax3 = C_ymax3[C];

                    mreal d1 = mymax(0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1));
                    mreal d2 = mymax(0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2));
                    mreal d3 = mymax(0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3));

                    mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;

                    if (h2 < theta2 * R2)
                    {
                        mreal b = C_B[C];
                        mreal y1 = C_Y1[C];
                        mreal y2 = C_Y2[C];
                        mreal y3 = C_Y3[C];

                        mreal local_local_sum = 0.;

                        for (mint i = i_begin; i < i_end; ++i)
                        {
                            mreal a = P_A[i];
                            mreal x1 = P_X1[i];
                            mreal x2 = P_X2[i];
                            mreal x3 = P_X3[i];
                            mreal n1 = P_N1[i];
                            mreal n2 = P_N2[i];
                            mreal n3 = P_N3[i];

                            mreal v1 = y1 - x1;
                            mreal v2 = y2 - x2;
                            mreal v3 = y3 - x3;

                            mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                            mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;
                            local_local_sum += a * mypow(fabs(rCosPhi), alpha) * mypow(r2, minus_betahalf);
                        }
                        local_sum += local_local_sum * b;
                    }
                    else
                    {
                        mint left = C_left[C];
                        mint right = C_right[C];
                        if (left >= 0 && right >= 0)
                        {
                            stack->push_back(right);
                            stack->push_back(left);
                        }
                        else
                        {
                            // near field loop
                            mint j_begin = C_ybegin[C];
                            mint j_end = C_yend[C];

                            for (mint i = i_begin; i < i_end; ++i)
                            {
                                mreal a = P_A[i];
                                mreal x1 = P_X1[i];
                                mreal x2 = P_X2[i];
                                mreal x3 = P_X3[i];
                                mreal n1 = P_N1[i];
                                mreal n2 = P_N2[i];
                                mreal n3 = P_N3[i];

                                mreal local_local_sum = 0.;

                                //                        #pragma omp simd aligned( P_A, P_X1, P_X3 : ALIGN )
                                for (mint j = j_begin; j < j_end; ++j)
                                {
                                    mreal b = P_B[j];
                                    mreal v1 = P_Y1[j] - x1;
                                    mreal v2 = P_Y2[j] - x2;
                                    mreal v3 = P_Y3[j] - x3;

                                    mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                                    mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;

                                    local_local_sum += mypow(fabs(rCosPhi), alpha) * mypow(r2, minus_betahalf) * b;
                                }

                                local_sum += a * local_local_sum;
                            }
                        }
                    }
                }

                sum += local_sum;
            }
        }

        {
            auto S = o_bvh;
            auto T = bvh;
            mreal const *const restrict C_xmin1 = S->C_min[0];
            mreal const *const restrict C_xmin2 = S->C_min[1];
            mreal const *const restrict C_xmin3 = S->C_min[2];
            mreal const *const restrict C_xmax1 = S->C_max[0];
            mreal const *const restrict C_xmax2 = S->C_max[1];
            mreal const *const restrict C_xmax3 = S->C_max[2];

            mreal const *const restrict C_xr2 = S->C_squared_radius;

            mreal const *const restrict P_A = S->P_data[0];
            mreal const *const restrict P_X1 = S->P_data[1];
            mreal const *const restrict P_X2 = S->P_data[2];
            mreal const *const restrict P_X3 = S->P_data[3];
            mreal const *const restrict P_N1 = S->P_data[4];
            mreal const *const restrict P_N2 = S->P_data[5];
            mreal const *const restrict P_N3 = S->P_data[6];

            mint const *const restrict C_xbegin = S->C_begin;
            mint const *const restrict C_xend = S->C_end;

            mint const *const restrict leaf = S->leaf_clusters;

            mreal const *const restrict C_ymin1 = T->C_min[0];
            mreal const *const restrict C_ymin2 = T->C_min[1];
            mreal const *const restrict C_ymin3 = T->C_min[2];
            mreal const *const restrict C_ymax1 = T->C_max[0];
            mreal const *const restrict C_ymax2 = T->C_max[1];
            mreal const *const restrict C_ymax3 = T->C_max[2];

            mreal const *const restrict C_yr2 = T->C_squared_radius;

            mreal const *const restrict P_B = T->P_data[0];
            mreal const *const restrict P_Y1 = T->P_data[1];
            mreal const *const restrict P_Y2 = T->P_data[2];
            mreal const *const restrict P_Y3 = T->P_data[3];

            mreal const *const restrict C_B = T->C_data[0];
            mreal const *const restrict C_Y1 = T->C_data[1];
            mreal const *const restrict C_Y2 = T->C_data[2];
            mreal const *const restrict C_Y3 = T->C_data[3];

            mint const *const restrict C_ybegin = T->C_begin;
            mint const *const restrict C_yend = T->C_end;

            mint const *const restrict C_left = T->C_left;
            mint const *const restrict C_right = T->C_right;

            A_Vector<A_Vector<mint>> thread_stack(nthreads);

#pragma omp parallel for num_threads(nthreads) reduction(+ \
                                                         : sum)
            for (mint k = 0; k < S->leaf_cluster_count; ++k)
            {
                mint thread = omp_get_thread_num();

                A_Vector<mint> *stack = &thread_stack[thread];

                stack->clear();
                stack->push_back(0);

                mint l = leaf[k];
                mint i_begin = C_xbegin[l];
                mint i_end = C_xend[l];

                mreal xmin1 = C_xmin1[l];
                mreal xmin2 = C_xmin2[l];
                mreal xmin3 = C_xmin3[l];

                mreal xmax1 = C_xmax1[l];
                mreal xmax2 = C_xmax2[l];
                mreal xmax3 = C_xmax3[l];

                mreal r2l = C_xr2[l];

                mreal local_sum = 0.;

                while (!stack->empty())
                {
                    mint C = stack->back();
                    stack->pop_back();

                    mreal h2 = std::max(r2l, C_yr2[C]);

                    // Compute squared distance between bounding boxes.
                    // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares

                    mreal ymin1 = C_ymin1[C];
                    mreal ymin2 = C_ymin2[C];
                    mreal ymin3 = C_ymin3[C];

                    mreal ymax1 = C_ymax1[C];
                    mreal ymax2 = C_ymax2[C];
                    mreal ymax3 = C_ymax3[C];

                    mreal d1 = mymax(0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1));
                    mreal d2 = mymax(0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2));
                    mreal d3 = mymax(0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3));

                    mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;

                    if (h2 < theta2 * R2)
                    {
                        mreal b = C_B[C];
                        mreal y1 = C_Y1[C];
                        mreal y2 = C_Y2[C];
                        mreal y3 = C_Y3[C];

                        mreal local_local_sum = 0.;

                        for (mint i = i_begin; i < i_end; ++i)
                        {
                            mreal a = P_A[i];
                            mreal x1 = P_X1[i];
                            mreal x2 = P_X2[i];
                            mreal x3 = P_X3[i];
                            mreal n1 = P_N1[i];
                            mreal n2 = P_N2[i];
                            mreal n3 = P_N3[i];

                            mreal v1 = y1 - x1;
                            mreal v2 = y2 - x2;
                            mreal v3 = y3 - x3;

                            mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                            mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;
                            local_local_sum += a * mypow(fabs(rCosPhi), alpha) * mypow(r2, minus_betahalf);
                        }
                        local_sum += local_local_sum * b;
                    }
                    else
                    {
                        mint left = C_left[C];
                        mint right = C_right[C];
                        if (left >= 0 && right >= 0)
                        {
                            stack->push_back(right);
                            stack->push_back(left);
                        }
                        else
                        {
                            // near field loop
                            mint j_begin = C_ybegin[C];
                            mint j_end = C_yend[C];

                            for (mint i = i_begin; i < i_end; ++i)
                            {
                                mreal a = P_A[i];
                                mreal x1 = P_X1[i];
                                mreal x2 = P_X2[i];
                                mreal x3 = P_X3[i];
                                mreal n1 = P_N1[i];
                                mreal n2 = P_N2[i];
                                mreal n3 = P_N3[i];

                                mreal local_local_sum = 0.;

                                //                        #pragma omp simd aligned( P_A, P_X1, P_X3 : ALIGN )
                                for (mint j = j_begin; j < j_end; ++j)
                                {
                                    mreal b = P_B[j];
                                    mreal v1 = P_Y1[j] - x1;
                                    mreal v2 = P_Y2[j] - x2;
                                    mreal v3 = P_Y3[j] - x3;

                                    mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                                    mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;

                                    local_local_sum += mypow(fabs(rCosPhi), alpha) * mypow(r2, minus_betahalf) * b;
                                }

                                local_sum += a * local_local_sum;
                            }
                        }
                    }
                }

                sum += local_sum;
            }
        }

        return sum;
    }; //Energy

    template <typename T1, typename T2>
    mreal TPObstacleBarnesHut_Projectors0::DEnergy(T1 alpha, T2 betahalf)
    {

        T1 alpha_minus_2 = alpha - 2;
        T2 minus_betahalf_minus_1 = -betahalf - 1;

        mreal beta = 2. * betahalf;
        mreal theta2 = theta * theta;
        mreal sum = 0.;

        mint data_dim = bvh->data_dim;
        mint nthreads = bvh->thread_count;

        {
            auto S = bvh;
            auto T = o_bvh;
            mreal const *const restrict C_xmin1 = S->C_min[0];
            mreal const *const restrict C_xmin2 = S->C_min[1];
            mreal const *const restrict C_xmin3 = S->C_min[2];
            mreal const *const restrict C_xmax1 = S->C_max[0];
            mreal const *const restrict C_xmax2 = S->C_max[1];
            mreal const *const restrict C_xmax3 = S->C_max[2];

            mreal const *const restrict C_xr2 = S->C_squared_radius;

            mreal const *const restrict P_A = S->P_data[0];
            mreal const *const restrict P_X1 = S->P_data[1];
            mreal const *const restrict P_X2 = S->P_data[2];
            mreal const *const restrict P_X3 = S->P_data[3];
            mreal const *const restrict P_N1 = S->P_data[4];
            mreal const *const restrict P_N2 = S->P_data[5];
            mreal const *const restrict P_N3 = S->P_data[6];

            mint const *const restrict C_xbegin = S->C_begin;
            mint const *const restrict C_xend = S->C_end;

            mint const *const restrict leaf = S->leaf_clusters;

            mreal const *const restrict C_ymin1 = T->C_min[0];
            mreal const *const restrict C_ymin2 = T->C_min[1];
            mreal const *const restrict C_ymin3 = T->C_min[2];
            mreal const *const restrict C_ymax1 = T->C_max[0];
            mreal const *const restrict C_ymax2 = T->C_max[1];
            mreal const *const restrict C_ymax3 = T->C_max[2];

            mreal const *const restrict C_yr2 = T->C_squared_radius;

            mreal const *const restrict P_B = T->P_data[0];
            mreal const *const restrict P_Y1 = T->P_data[1];
            mreal const *const restrict P_Y2 = T->P_data[2];
            mreal const *const restrict P_Y3 = T->P_data[3];

            mreal const *const restrict C_B = T->C_data[0];
            mreal const *const restrict C_Y1 = T->C_data[1];
            mreal const *const restrict C_Y2 = T->C_data[2];
            mreal const *const restrict C_Y3 = T->C_data[3];

            mint const *const restrict C_ybegin = T->C_begin;
            mint const *const restrict C_yend = T->C_end;

            mint const *const restrict C_left = T->C_left;
            mint const *const restrict C_right = T->C_right;

            A_Vector<A_Vector<mint>> thread_stack(nthreads);

#pragma omp parallel for num_threads(nthreads) reduction(+ \
                                                         : sum)
            for (mint k = 0; k < S->leaf_cluster_count; ++k)
            {
                mint thread = omp_get_thread_num();

                A_Vector<mint> *stack = &thread_stack[thread];

                mreal *const restrict P_U = &S->P_D_data[thread][0];
                //            mreal * const restrict P_V = &T->P_D_data[thread][0];
                //            mreal * const restrict C_V = &T->C_D_data[thread][0];

                stack->clear();
                stack->push_back(0);

                mint l = leaf[k];
                mint i_begin = C_xbegin[l];
                mint i_end = C_xend[l];

                mreal xmin1 = C_xmin1[l];
                mreal xmin2 = C_xmin2[l];
                mreal xmin3 = C_xmin3[l];

                mreal xmax1 = C_xmax1[l];
                mreal xmax2 = C_xmax2[l];
                mreal xmax3 = C_xmax3[l];

                mreal r2l = C_xr2[l];

                while (!stack->empty())
                {
                    mint C = stack->back();
                    stack->pop_back();

                    mreal h2 = std::max(r2l, C_yr2[C]);

                    // Compute squared distance between bounding boxes.
                    // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares

                    mreal ymin1 = C_ymin1[C];
                    mreal ymin2 = C_ymin2[C];
                    mreal ymin3 = C_ymin3[C];

                    mreal ymax1 = C_ymax1[C];
                    mreal ymax2 = C_ymax2[C];
                    mreal ymax3 = C_ymax3[C];

                    mreal d1 = mymax(0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1));
                    mreal d2 = mymax(0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2));
                    mreal d3 = mymax(0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3));

                    mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;

                    if (h2 < theta2 * R2)
                    {
                        mreal b = C_B[C];
                        mreal y1 = C_Y1[C];
                        mreal y2 = C_Y2[C];
                        mreal y3 = C_Y3[C];

                        for (mint i = i_begin; i < i_end; ++i)
                        {
                            mreal a = P_A[i];
                            mreal x1 = P_X1[i];
                            mreal x2 = P_X2[i];
                            mreal x3 = P_X3[i];
                            mreal n1 = P_N1[i];
                            mreal n2 = P_N2[i];
                            mreal n3 = P_N3[i];

                            mreal v1 = y1 - x1;
                            mreal v2 = y2 - x2;
                            mreal v3 = y3 - x3;

                            mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                            mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;

                            mreal rBetaMinus2 = mypow(r2, minus_betahalf_minus_1);
                            mreal rBeta = rBetaMinus2 * r2;

                            mreal rCosPhiAlphaMinus1 = mypow(fabs(rCosPhi), alpha_minus_2) * rCosPhi;
                            mreal rCosPhiAlpha = rCosPhiAlphaMinus1 * rCosPhi;

                            mreal Num = rCosPhiAlpha;
                            mreal factor0 = rBeta * alpha;
                            mreal density = rBeta * Num;
                            sum += a * b * density;

                            mreal F = factor0 * rCosPhiAlphaMinus1;
                            mreal H = beta * rBetaMinus2 * Num;

                            mreal bF = b * F;

                            mreal Z1 = (-n1 * F + v1 * H);
                            mreal Z2 = (-n2 * F + v2 * H);
                            mreal Z3 = (-n3 * F + v3 * H);

                            P_U[data_dim * i] += b * (density +
                                                      F * (n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3)) -
                                                      H * (v1 * x1 + v2 * x2 + v3 * x3));
                            P_U[data_dim * i + 1] += b * Z1;
                            P_U[data_dim * i + 2] += b * Z2;
                            P_U[data_dim * i + 3] += b * Z3;
                            P_U[data_dim * i + 4] += bF * v1;
                            P_U[data_dim * i + 5] += bF * v2;
                            P_U[data_dim * i + 6] += bF * v3;

                            //                        C_V[ 7 * C + 0 ] += a * (
                            //                                                 density
                            //                                                 -
                            //                                                 F * ( n1 * y1 + n2 * y2 + n3 * y3 )
                            //                                                 +
                            //                                                 H * ( v1 * y1 + v2 * y2 + v3 * y3 )
                            //                                                 );
                            //                        C_V[ 7 * C + 1 ] -= a  * Z1;
                            //                        C_V[ 7 * C + 2 ] -= a  * Z2;
                            //                        C_V[ 7 * C + 3 ] -= a  * Z3;
                        }
                    }
                    else
                    {
                        mint left = C_left[C];
                        mint right = C_right[C];
                        if (left >= 0 && right >= 0)
                        {
                            stack->push_back(right);
                            stack->push_back(left);
                        }
                        else
                        {
                            // near field loop
                            mint j_begin = C_ybegin[C];
                            mint j_end = C_yend[C];

                            for (mint i = i_begin; i < i_end; ++i)
                            {
                                mreal a = P_A[i];
                                mreal x1 = P_X1[i];
                                mreal x2 = P_X2[i];
                                mreal x3 = P_X3[i];
                                mreal n1 = P_N1[i];
                                mreal n2 = P_N2[i];
                                mreal n3 = P_N3[i];

                                mreal da = 0.;
                                mreal dx1 = 0.;
                                mreal dx2 = 0.;
                                mreal dx3 = 0.;
                                mreal dn1 = 0.;
                                mreal dn2 = 0.;
                                mreal dn3 = 0.;

#pragma omp simd aligned(P_B, P_Y1, P_Y2, P_Y3 \
                         : ALIGN) reduction(+  \
                                            : sum)
                                for (mint j = j_begin; j < j_end; ++j)
                                {
                                    mreal b = P_B[j];
                                    mreal y1 = P_Y1[j];
                                    mreal y2 = P_Y2[j];
                                    mreal y3 = P_Y3[j];

                                    mreal v1 = y1 - x1;
                                    mreal v2 = y2 - x2;
                                    mreal v3 = y3 - x3;

                                    mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                                    mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;

                                    mreal rBetaMinus2 = mypow(r2, minus_betahalf_minus_1);
                                    mreal rBeta = rBetaMinus2 * r2;

                                    mreal rCosPhiAlphaMinus1 = mypow(fabs(rCosPhi), alpha_minus_2) * rCosPhi;
                                    mreal rCosPhiAlpha = rCosPhiAlphaMinus1 * rCosPhi;

                                    mreal Num = rCosPhiAlpha;
                                    mreal factor0 = rBeta * alpha;
                                    mreal density = rBeta * Num;
                                    sum += a * b * density;

                                    mreal F = factor0 * rCosPhiAlphaMinus1;
                                    mreal H = beta * rBetaMinus2 * Num;

                                    mreal bF = b * F;

                                    mreal Z1 = (-n1 * F + v1 * H);
                                    mreal Z2 = (-n2 * F + v2 * H);
                                    mreal Z3 = (-n3 * F + v3 * H);

                                    da += b * (density +
                                               F * (n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3)) -
                                               H * (v1 * x1 + v2 * x2 + v3 * x3));
                                    dx1 += b * Z1;
                                    dx2 += b * Z2;
                                    dx3 += b * Z3;
                                    dn1 += bF * v1;
                                    dn2 += bF * v2;
                                    dn3 += bF * v3;

                                    //                                P_V[ 7 * j + 0 ] += a * (
                                    //                                                         density
                                    //                                                         -
                                    //                                                         F * ( n1 * y1 + n2 * y2 + n3 * y3 )
                                    //                                                         +
                                    //                                                         H * ( v1 * y1 + v2 * y2 + v3 * y3 )
                                    //                                                         );
                                    //                                P_V[ 7 * j + 1 ] -= a  * Z1;
                                    //                                P_V[ 7 * j + 2 ] -= a  * Z2;
                                    //                                P_V[ 7 * j + 3 ] -= a  * Z3;
                                }

                                P_U[data_dim * i] += da;
                                P_U[data_dim * i + 1] += dx1;
                                P_U[data_dim * i + 2] += dx2;
                                P_U[data_dim * i + 3] += dx3;
                                P_U[data_dim * i + 4] += dn1;
                                P_U[data_dim * i + 5] += dn2;
                                P_U[data_dim * i + 6] += dn3;
                            }
                        }
                    }
                }
            }
        }

        {
            auto S = o_bvh;
            auto T = bvh;
            mreal const *const restrict C_xmin1 = S->C_min[0];
            mreal const *const restrict C_xmin2 = S->C_min[1];
            mreal const *const restrict C_xmin3 = S->C_min[2];
            mreal const *const restrict C_xmax1 = S->C_max[0];
            mreal const *const restrict C_xmax2 = S->C_max[1];
            mreal const *const restrict C_xmax3 = S->C_max[2];

            mreal const *const restrict C_xr2 = S->C_squared_radius;

            mreal const *const restrict P_A = S->P_data[0];
            mreal const *const restrict P_X1 = S->P_data[1];
            mreal const *const restrict P_X2 = S->P_data[2];
            mreal const *const restrict P_X3 = S->P_data[3];
            mreal const *const restrict P_N1 = S->P_data[4];
            mreal const *const restrict P_N2 = S->P_data[5];
            mreal const *const restrict P_N3 = S->P_data[6];

            mint const *const restrict C_xbegin = S->C_begin;
            mint const *const restrict C_xend = S->C_end;

            mint const *const restrict leaf = S->leaf_clusters;

            mreal const *const restrict C_ymin1 = T->C_min[0];
            mreal const *const restrict C_ymin2 = T->C_min[1];
            mreal const *const restrict C_ymin3 = T->C_min[2];
            mreal const *const restrict C_ymax1 = T->C_max[0];
            mreal const *const restrict C_ymax2 = T->C_max[1];
            mreal const *const restrict C_ymax3 = T->C_max[2];

            mreal const *const restrict C_yr2 = T->C_squared_radius;

            mreal const *const restrict P_B = T->P_data[0];
            mreal const *const restrict P_Y1 = T->P_data[1];
            mreal const *const restrict P_Y2 = T->P_data[2];
            mreal const *const restrict P_Y3 = T->P_data[3];

            mreal const *const restrict C_B = T->C_data[0];
            mreal const *const restrict C_Y1 = T->C_data[1];
            mreal const *const restrict C_Y2 = T->C_data[2];
            mreal const *const restrict C_Y3 = T->C_data[3];

            mint const *const restrict C_ybegin = T->C_begin;
            mint const *const restrict C_yend = T->C_end;

            mint const *const restrict C_left = T->C_left;
            mint const *const restrict C_right = T->C_right;

            A_Vector<A_Vector<mint>> thread_stack(nthreads);

#pragma omp parallel for num_threads(nthreads) reduction(+ \
                                                         : sum)
            for (mint k = 0; k < S->leaf_cluster_count; ++k)
            {
                mint thread = omp_get_thread_num();

                A_Vector<mint> *stack = &thread_stack[thread];

                //            mreal * const restrict P_U = &S->P_D_data[thread][0];
                mreal *const restrict P_V = &T->P_D_data[thread][0];
                mreal *const restrict C_V = &T->C_D_data[thread][0];

                stack->clear();
                stack->push_back(0);

                mint l = leaf[k];
                mint i_begin = C_xbegin[l];
                mint i_end = C_xend[l];

                mreal xmin1 = C_xmin1[l];
                mreal xmin2 = C_xmin2[l];
                mreal xmin3 = C_xmin3[l];

                mreal xmax1 = C_xmax1[l];
                mreal xmax2 = C_xmax2[l];
                mreal xmax3 = C_xmax3[l];

                mreal r2l = C_xr2[l];

                while (!stack->empty())
                {
                    mint C = stack->back();
                    stack->pop_back();

                    mreal h2 = std::max(r2l, C_yr2[C]);

                    // Compute squared distance between bounding boxes.
                    // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares

                    mreal ymin1 = C_ymin1[C];
                    mreal ymin2 = C_ymin2[C];
                    mreal ymin3 = C_ymin3[C];

                    mreal ymax1 = C_ymax1[C];
                    mreal ymax2 = C_ymax2[C];
                    mreal ymax3 = C_ymax3[C];

                    mreal d1 = mymax(0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1));
                    mreal d2 = mymax(0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2));
                    mreal d3 = mymax(0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3));

                    mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;

                    if (h2 < theta2 * R2)
                    {
                        mreal b = C_B[C];
                        mreal y1 = C_Y1[C];
                        mreal y2 = C_Y2[C];
                        mreal y3 = C_Y3[C];

                        mreal db = 0.;
                        mreal dy1 = 0.;
                        mreal dy2 = 0.;
                        mreal dy3 = 0.;

#pragma omp simd aligned(P_A, P_X1, P_X2, P_X3, P_N1, P_N2, P_N3 \
                         : ALIGN) reduction(+                    \
                                            : sum)
                        for (mint i = i_begin; i < i_end; ++i)
                        {
                            mreal a = P_A[i];
                            mreal x1 = P_X1[i];
                            mreal x2 = P_X2[i];
                            mreal x3 = P_X3[i];
                            mreal n1 = P_N1[i];
                            mreal n2 = P_N2[i];
                            mreal n3 = P_N3[i];

                            mreal v1 = y1 - x1;
                            mreal v2 = y2 - x2;
                            mreal v3 = y3 - x3;

                            mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                            mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;

                            mreal rBetaMinus2 = mypow(r2, minus_betahalf_minus_1);
                            mreal rBeta = rBetaMinus2 * r2;

                            mreal rCosPhiAlphaMinus1 = mypow(fabs(rCosPhi), alpha_minus_2) * rCosPhi;
                            mreal rCosPhiAlpha = rCosPhiAlphaMinus1 * rCosPhi;

                            mreal Num = rCosPhiAlpha;
                            mreal factor0 = rBeta * alpha;
                            mreal density = rBeta * Num;
                            sum += a * b * density;

                            mreal F = factor0 * rCosPhiAlphaMinus1;
                            mreal H = beta * rBetaMinus2 * Num;

                            mreal bF = b * F;

                            mreal Z1 = (-n1 * F + v1 * H);
                            mreal Z2 = (-n2 * F + v2 * H);
                            mreal Z3 = (-n3 * F + v3 * H);

                            //                        P_U[ data_dim * i     ] += b * (
                            //                                   density
                            //                                   +
                            //                                   F * ( n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3) )
                            //                                   -
                            //                                   H * ( v1 * x1 + v2 * x2 + v3 * x3 )
                            //                                   );
                            //                        P_U[ data_dim * i + 1 ] += b  * Z1;
                            //                        P_U[ data_dim * i + 2 ] += b  * Z2;
                            //                        P_U[ data_dim * i + 3 ] += b  * Z3;
                            //                        P_U[ data_dim * i + 4 ] += bF * v1;
                            //                        P_U[ data_dim * i + 5 ] += bF * v2;
                            //                        P_U[ data_dim * i + 6 ] += bF * v3;

                            db += a * (density -
                                       F * (n1 * y1 + n2 * y2 + n3 * y3) +
                                       H * (v1 * y1 + v2 * y2 + v3 * y3));
                            dy1 -= a * Z1;
                            dy2 -= a * Z2;
                            dy3 -= a * Z3;
                        }
                        C_V[7 * C + 0] += db;
                        C_V[7 * C + 1] += dy1;
                        C_V[7 * C + 2] += dy2;
                        C_V[7 * C + 3] += dy3;
                    }
                    else
                    {
                        mint left = C_left[C];
                        mint right = C_right[C];
                        if (left >= 0 && right >= 0)
                        {
                            stack->push_back(right);
                            stack->push_back(left);
                        }
                        else
                        {
                            // near field loop
                            mint j_begin = C_ybegin[C];
                            mint j_end = C_yend[C];

                            for (mint i = i_begin; i < i_end; ++i)
                            {
                                mreal a = P_A[i];
                                mreal x1 = P_X1[i];
                                mreal x2 = P_X2[i];
                                mreal x3 = P_X3[i];
                                mreal n1 = P_N1[i];
                                mreal n2 = P_N2[i];
                                mreal n3 = P_N3[i];

                                //                            mreal  da = 0.;
                                //                            mreal dx1 = 0.;
                                //                            mreal dx2 = 0.;
                                //                            mreal dx3 = 0.;
                                //                            mreal dn1 = 0.;
                                //                            mreal dn2 = 0.;
                                //                            mreal dn3 = 0.;

#pragma omp simd aligned(P_B, P_Y1, P_Y2, P_Y3, P_V \
                         : ALIGN) reduction(+       \
                                            : sum)
                                for (mint j = j_begin; j < j_end; ++j)
                                {
                                    mreal b = P_B[j];
                                    mreal y1 = P_Y1[j];
                                    mreal y2 = P_Y2[j];
                                    mreal y3 = P_Y3[j];

                                    mreal v1 = y1 - x1;
                                    mreal v2 = y2 - x2;
                                    mreal v3 = y3 - x3;

                                    mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                                    mreal r2 = v1 * v1 + v2 * v2 + v3 * v3;

                                    mreal rBetaMinus2 = mypow(r2, minus_betahalf_minus_1);
                                    mreal rBeta = rBetaMinus2 * r2;

                                    mreal rCosPhiAlphaMinus1 = mypow(fabs(rCosPhi), alpha_minus_2) * rCosPhi;
                                    mreal rCosPhiAlpha = rCosPhiAlphaMinus1 * rCosPhi;

                                    mreal Num = rCosPhiAlpha;
                                    mreal factor0 = rBeta * alpha;
                                    mreal density = rBeta * Num;
                                    sum += a * b * density;

                                    mreal F = factor0 * rCosPhiAlphaMinus1;
                                    mreal H = beta * rBetaMinus2 * Num;

                                    mreal bF = b * F;

                                    mreal Z1 = (-n1 * F + v1 * H);
                                    mreal Z2 = (-n2 * F + v2 * H);
                                    mreal Z3 = (-n3 * F + v3 * H);

                                    //                                da += b * (
                                    //                                           density
                                    //                                           +
                                    //                                           F * ( n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3) )
                                    //                                           -
                                    //                                           H * ( v1 * x1 + v2 * x2 + v3 * x3 )
                                    //                                           );
                                    //                                dx1 += b  * Z1;
                                    //                                dx2 += b  * Z2;
                                    //                                dx3 += b  * Z3;
                                    //                                dn1 += bF * v1;
                                    //                                dn2 += bF * v2;
                                    //                                dn3 += bF * v3;

                                    P_V[7 * j + 0] += a * (density -
                                                           F * (n1 * y1 + n2 * y2 + n3 * y3) +
                                                           H * (v1 * y1 + v2 * y2 + v3 * y3));
                                    P_V[7 * j + 1] -= a * Z1;
                                    P_V[7 * j + 2] -= a * Z2;
                                    P_V[7 * j + 3] -= a * Z3;
                                }
                                //                            P_U[ data_dim * i     ] +=  da;
                                //                            P_U[ data_dim * i + 1 ] += dx1;
                                //                            P_U[ data_dim * i + 2 ] += dx2;
                                //                            P_U[ data_dim * i + 3 ] += dx3;
                                //                            P_U[ data_dim * i + 4 ] += dn1;
                                //                            P_U[ data_dim * i + 5 ] += dn2;
                                //                            P_U[ data_dim * i + 6 ] += dn3;
                            }
                        }
                    }
                }
            }
        }

        return sum;
    }; // DEnergy
} // namespace rsurfaces
