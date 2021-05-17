#include "energy/tp_obstacle_barnes_hut_0.h"

namespace rsurfaces
{
 
    template <typename T1, typename T2>
    mreal TPObstacleBarnesHut0::Energy(T1 alpha, T2 betahalf)
    {
        ptic("TPObstacleBarnesHut0::Energy");
        
        T2 minus_betahalf = -betahalf;
        mreal theta2 = theta * theta;

        mint nthreads = bvh->thread_count;
        
        mreal sum = 0.;
        {
            auto S = bvh;
            auto T = o_bvh;
            mreal const * restrict const  C_xmin1 = S->C_min[0];
            mreal const * restrict const  C_xmin2 = S->C_min[1];
            mreal const * restrict const  C_xmin3 = S->C_min[2];
            mreal const * restrict const  C_xmax1 = S->C_max[0];
            mreal const * restrict const  C_xmax2 = S->C_max[1];
            mreal const * restrict const  C_xmax3 = S->C_max[2];

            mreal const * restrict const  C_xr2 = S->C_squared_radius;

            mreal const * restrict const  P_A = S->P_near[0];
            mreal const * restrict const  P_X1 = S->P_near[1];
            mreal const * restrict const  P_X2 = S->P_near[2];
            mreal const * restrict const  P_X3 = S->P_near[3];
            mreal const * restrict const  P_N1 = S->P_near[4];
            mreal const * restrict const  P_N2 = S->P_near[5];
            mreal const * restrict const  P_N3 = S->P_near[6];

            mint  const * restrict const  C_xbegin = S->C_begin;
            mint  const * restrict const  C_xend = S->C_end;

            mint  const * restrict const  leaf = S->leaf_clusters;

            mreal const * restrict const  C_ymin1 = T->C_min[0];
            mreal const * restrict const  C_ymin2 = T->C_min[1];
            mreal const * restrict const  C_ymin3 = T->C_min[2];
            mreal const * restrict const  C_ymax1 = T->C_max[0];
            mreal const * restrict const  C_ymax2 = T->C_max[1];
            mreal const * restrict const  C_ymax3 = T->C_max[2];

            mreal const * restrict const  C_yr2 = T->C_squared_radius;

            mreal const * restrict const  P_B = T->P_near[0];
            mreal const * restrict const  P_Y1 = T->P_near[1];
            mreal const * restrict const  P_Y2 = T->P_near[2];
            mreal const * restrict const  P_Y3 = T->P_near[3];

            mreal const * restrict const  C_B = T->C_far[0];
            mreal const * restrict const  C_Y1 = T->C_far[1];
            mreal const * restrict const  C_Y2 = T->C_far[2];
            mreal const * restrict const  C_Y3 = T->C_far[3];

            mint  const * restrict const  C_ybegin = T->C_begin;
            mint  const * restrict const  C_yend = T->C_end;

            mint  const * restrict const  C_left = T->C_left;
            mint  const * restrict const  C_right = T->C_right;

            A_Vector<A_Vector<mint>> thread_stack(nthreads);

            #pragma omp parallel for num_threads(nthreads) reduction(+ : sum)
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

                    mreal R2 = SquaredBoxMinDistance( xmin1,      xmin2,      xmin3,      xmax1,      xmax2,      xmax3,
                                                      C_ymin1[C], C_ymin2[C], C_ymin3[C], C_ymax1[C], C_ymax2[C], C_ymax3[C]);
                    
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
        
        if( two_sided )
        {
            auto S = o_bvh;
            auto T = bvh;
            mreal const * restrict const  C_xmin1 = S->C_min[0];
            mreal const * restrict const  C_xmin2 = S->C_min[1];
            mreal const * restrict const  C_xmin3 = S->C_min[2];
            mreal const * restrict const  C_xmax1 = S->C_max[0];
            mreal const * restrict const  C_xmax2 = S->C_max[1];
            mreal const * restrict const  C_xmax3 = S->C_max[2];

            mreal const * restrict const  C_xr2 = S->C_squared_radius;

            mreal const * restrict const  P_A = S->P_near[0];
            mreal const * restrict const  P_X1 = S->P_near[1];
            mreal const * restrict const  P_X2 = S->P_near[2];
            mreal const * restrict const  P_X3 = S->P_near[3];
            mreal const * restrict const  P_N1 = S->P_near[4];
            mreal const * restrict const  P_N2 = S->P_near[5];
            mreal const * restrict const  P_N3 = S->P_near[6];

            mint  const * restrict const  C_xbegin = S->C_begin;
            mint  const * restrict const  C_xend = S->C_end;

            mint  const * restrict const  leaf = S->leaf_clusters;

            mreal const * restrict const  C_ymin1 = T->C_min[0];
            mreal const * restrict const  C_ymin2 = T->C_min[1];
            mreal const * restrict const  C_ymin3 = T->C_min[2];
            mreal const * restrict const  C_ymax1 = T->C_max[0];
            mreal const * restrict const  C_ymax2 = T->C_max[1];
            mreal const * restrict const  C_ymax3 = T->C_max[2];

            mreal const * restrict const  C_yr2 = T->C_squared_radius;

            mreal const * restrict const  P_B = T->P_near[0];
            mreal const * restrict const  P_Y1 = T->P_near[1];
            mreal const * restrict const  P_Y2 = T->P_near[2];
            mreal const * restrict const  P_Y3 = T->P_near[3];

            mreal const * restrict const  C_B = T->C_far[0];
            mreal const * restrict const  C_Y1 = T->C_far[1];
            mreal const * restrict const  C_Y2 = T->C_far[2];
            mreal const * restrict const  C_Y3 = T->C_far[3];

            mint  const * restrict const  C_ybegin = T->C_begin;
            mint  const * restrict const  C_yend = T->C_end;

            mint  const * restrict const  C_left = T->C_left;
            mint  const * restrict const  C_right = T->C_right;

            A_Vector<A_Vector<mint>> thread_stack(nthreads);

            #pragma omp parallel for num_threads(nthreads) reduction(+ : sum)
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

                    mreal R2 = SquaredBoxMinDistance( xmin1,      xmin2,      xmin3,      xmax1,      xmax2,      xmax3,
                                                      C_ymin1[C], C_ymin2[C], C_ymin3[C], C_ymax1[C], C_ymax2[C], C_ymax3[C]);

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

        ptoc("TPObstacleBarnesHut0::Energy");
        
        return sum;
    }; //Energy

    template <typename T1, typename T2>
    mreal TPObstacleBarnesHut0::DEnergy(T1 alpha, T2 betahalf)
    {
        ptic("TPObstacleBarnesHut0::DEnergy");
        
        T1 alpha_minus_2 = alpha - 2;
        T2 minus_betahalf_minus_1 = -betahalf - 1;

        mreal beta = 2. * betahalf;
        mreal theta2 = theta * theta;
        mreal sum = 0.;
        
        mint nthreads = bvh->thread_count;

        {
            auto S = bvh;
            auto T = o_bvh;
            mint far_dim = S->far_dim;
            mreal const * restrict const  C_xmin1 = S->C_min[0];
            mreal const * restrict const  C_xmin2 = S->C_min[1];
            mreal const * restrict const  C_xmin3 = S->C_min[2];
            mreal const * restrict const  C_xmax1 = S->C_max[0];
            mreal const * restrict const  C_xmax2 = S->C_max[1];
            mreal const * restrict const  C_xmax3 = S->C_max[2];

            mreal const * restrict const  C_xr2 = S->C_squared_radius;

            mreal const * restrict const  P_A = S->P_near[0];
            mreal const * restrict const  P_X1 = S->P_near[1];
            mreal const * restrict const  P_X2 = S->P_near[2];
            mreal const * restrict const  P_X3 = S->P_near[3];
            mreal const * restrict const  P_N1 = S->P_near[4];
            mreal const * restrict const  P_N2 = S->P_near[5];
            mreal const * restrict const  P_N3 = S->P_near[6];

            mint  const * restrict const  C_xbegin = S->C_begin;
            mint  const * restrict const  C_xend = S->C_end;

            mint  const * restrict const  leaf = S->leaf_clusters;

            mreal const * restrict const  C_ymin1 = T->C_min[0];
            mreal const * restrict const  C_ymin2 = T->C_min[1];
            mreal const * restrict const  C_ymin3 = T->C_min[2];
            mreal const * restrict const  C_ymax1 = T->C_max[0];
            mreal const * restrict const  C_ymax2 = T->C_max[1];
            mreal const * restrict const  C_ymax3 = T->C_max[2];

            mreal const * restrict const  C_yr2 = T->C_squared_radius;

            mreal const * restrict const  P_B = T->P_near[0];
            mreal const * restrict const  P_Y1 = T->P_near[1];
            mreal const * restrict const  P_Y2 = T->P_near[2];
            mreal const * restrict const  P_Y3 = T->P_near[3];

            mreal const * restrict const  C_B = T->C_far[0];
            mreal const * restrict const  C_Y1 = T->C_far[1];
            mreal const * restrict const  C_Y2 = T->C_far[2];
            mreal const * restrict const  C_Y3 = T->C_far[3];

            mint  const * restrict const  C_ybegin = T->C_begin;
            mint  const * restrict const  C_yend = T->C_end;

            mint  const * restrict const  C_left = T->C_left;
            mint  const * restrict const  C_right = T->C_right;

            A_Vector<A_Vector<mint>> thread_stack(nthreads);

            #pragma omp parallel for num_threads(nthreads) reduction(+ : sum)
            for (mint k = 0; k < S->leaf_cluster_count; ++k)
            {
                mint thread = omp_get_thread_num();

                A_Vector<mint> *stack = &thread_stack[thread];

                mreal * restrict const P_U = &S->P_D_near[thread][0];

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

                    mreal R2 = SquaredBoxMinDistance( xmin1,      xmin2,      xmin3,      xmax1,      xmax2,      xmax3,
                                                      C_ymin1[C], C_ymin2[C], C_ymin3[C], C_ymax1[C], C_ymax2[C], C_ymax3[C]);

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

                            P_U[7 * i] += b * (density +
                                               F * (n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3)) -
                                               H * (v1 * x1 + v2 * x2 + v3 * x3));
                            P_U[7 * i + 1] += b * Z1;
                            P_U[7 * i + 2] += b * Z2;
                            P_U[7 * i + 3] += b * Z3;
                            P_U[7 * i + 4] += bF * v1;
                            P_U[7 * i + 5] += bF * v2;
                            P_U[7 * i + 6] += bF * v3;
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

                                #pragma omp simd aligned(P_B, P_Y1, P_Y2, P_Y3 : ALIGN) reduction(+ : sum)
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
                                }

                                P_U[7 * i] += da;
                                P_U[7 * i + 1] += dx1;
                                P_U[7 * i + 2] += dx2;
                                P_U[7 * i + 3] += dx3;
                                P_U[7 * i + 4] += dn1;
                                P_U[7 * i + 5] += dn2;
                                P_U[7 * i + 6] += dn3;
                            }
                        }
                    }
                }
            }
        }

        if( two_sided )
        {
            auto S = o_bvh;
            auto T = bvh;
            mint far_dim = T->far_dim;
            mreal const * restrict const  C_xmin1 = S->C_min[0];
            mreal const * restrict const  C_xmin2 = S->C_min[1];
            mreal const * restrict const  C_xmin3 = S->C_min[2];
            mreal const * restrict const  C_xmax1 = S->C_max[0];
            mreal const * restrict const  C_xmax2 = S->C_max[1];
            mreal const * restrict const  C_xmax3 = S->C_max[2];
            
            mreal const * restrict const  C_xr2 = S->C_squared_radius;
            
            mreal const * restrict const  P_A = S->P_near[0];
            mreal const * restrict const  P_X1 = S->P_near[1];
            mreal const * restrict const  P_X2 = S->P_near[2];
            mreal const * restrict const  P_X3 = S->P_near[3];
            mreal const * restrict const  P_N1 = S->P_near[4];
            mreal const * restrict const  P_N2 = S->P_near[5];
            mreal const * restrict const  P_N3 = S->P_near[6];
            
            mint  const * restrict const  C_xbegin = S->C_begin;
            mint  const * restrict const  C_xend = S->C_end;
            
            mint  const * restrict const  leaf = S->leaf_clusters;
            
            mreal const * restrict const  C_ymin1 = T->C_min[0];
            mreal const * restrict const  C_ymin2 = T->C_min[1];
            mreal const * restrict const  C_ymin3 = T->C_min[2];
            mreal const * restrict const  C_ymax1 = T->C_max[0];
            mreal const * restrict const  C_ymax2 = T->C_max[1];
            mreal const * restrict const  C_ymax3 = T->C_max[2];
            
            mreal const * restrict const  C_yr2 = T->C_squared_radius;
            
            mreal const * restrict const  P_B = T->P_near[0];
            mreal const * restrict const  P_Y1 = T->P_near[1];
            mreal const * restrict const  P_Y2 = T->P_near[2];
            mreal const * restrict const  P_Y3 = T->P_near[3];
            
            mreal const * restrict const  C_B = T->C_far[0];
            mreal const * restrict const  C_Y1 = T->C_far[1];
            mreal const * restrict const  C_Y2 = T->C_far[2];
            mreal const * restrict const  C_Y3 = T->C_far[3];
            
            mint  const * restrict const  C_ybegin = T->C_begin;
            mint  const * restrict const  C_yend = T->C_end;
            
            mint  const * restrict const  C_left = T->C_left;
            mint  const * restrict const  C_right = T->C_right;
            
            A_Vector<A_Vector<mint>> thread_stack(nthreads);
            
            #pragma omp parallel for num_threads(nthreads) reduction(+ : sum)
            for (mint k = 0; k < S->leaf_cluster_count; ++k)
            {
                mint thread = omp_get_thread_num();
                
                A_Vector<mint> *stack = &thread_stack[thread];
                
                mreal * restrict const P_V = &T->P_D_near[thread][0];
                mreal * restrict const C_V = &T->C_D_far[thread][0];
                
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
                    
                    mreal R2 = SquaredBoxMinDistance( xmin1,      xmin2,      xmin3,      xmax1,      xmax2,      xmax3,
                                                      C_ymin1[C], C_ymin2[C], C_ymin3[C], C_ymax1[C], C_ymax2[C], C_ymax3[C]);
                    
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
                        
                        #pragma omp simd aligned(P_A, P_X1, P_X2, P_X3, P_N1, P_N2, P_N3 : ALIGN) reduction(+ : sum)
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
                            
                            
                            db += a * (density -
                                       F * (n1 * y1 + n2 * y2 + n3 * y3) +
                                       H * (v1 * y1 + v2 * y2 + v3 * y3));
                            dy1 -= a * Z1;
                            dy2 -= a * Z2;
                            dy3 -= a * Z3;
                        }
                        C_V[far_dim * C + 0] += db;
                        C_V[far_dim * C + 1] += dy1;
                        C_V[far_dim * C + 2] += dy2;
                        C_V[far_dim * C + 3] += dy3;
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
                                
                                #pragma omp simd aligned(P_B, P_Y1, P_Y2, P_Y3, P_V : ALIGN) reduction(+ : sum)
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
                                    
                                    P_V[7 * j + 0] += a * (density -
                                                           F * (n1 * y1 + n2 * y2 + n3 * y3) +
                                                           H * (v1 * y1 + v2 * y2 + v3 * y3));
                                    P_V[7 * j + 1] -= a * Z1;
                                    P_V[7 * j + 2] -= a * Z2;
                                    P_V[7 * j + 3] -= a * Z3;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        ptoc("TPObstacleBarnesHut0::DEnergy");
        
        return sum;
    }; // DEnergy
    
    double TPObstacleBarnesHut0::Value()
    {
        ptic("TPObstacleBarnesHut0::Value");
        
        mreal value = 0.;

        bvh = bvhSharedFrom->GetBVH();
        if (!bvh)
        {
            throw std::runtime_error("Obstacle energy is sharing BVH from an energy that has no BVH.");
        }
        if (use_int)
        {
            mint int_alpha = std::round(alpha);
            mint int_betahalf = std::round(beta / 2);
            value = weight * Energy(int_alpha, int_betahalf);
        }
        else
        {
            mreal real_alpha = alpha;
            mreal real_betahalf = beta / 2;
            value = weight * Energy(real_alpha, real_betahalf);
        }

        ptoc("TPObstacleBarnesHut0::Value");

        return value;
    } // Value

    void TPObstacleBarnesHut0::Differential(Eigen::MatrixXd &output)
    {
        ptic("TPObstacleBarnesHut0::Differential");
        
        bvh = bvhSharedFrom->GetBVH();
        if (!bvh)
        {
            throw std::runtime_error("Obstacle energy is sharing BVH from an energy that has no BVH.");
        }
        
        if( bvh->near_dim != 7)
        {
            eprint("in TPObstacleBarnesHut0::Differential: near_dim != 7");
        }
        
        EigenMatrixRM P_D_near_( bvh->primitive_count, bvh->near_dim );
        EigenMatrixRM P_D_far_ ( bvh->primitive_count, bvh->far_dim );

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

        bvh->CollectDerivatives( P_D_near_.data(), P_D_far_.data() );

        AssembleDerivativeFromACNData( mesh, geom, P_D_near_, output, weight );

        if( bvh->far_dim == 10)
        {
            AssembleDerivativeFromACPData( mesh, geom, P_D_far_, output, weight );
        }
        else
        {
            AssembleDerivativeFromACNData( mesh, geom, P_D_far_, output, weight );
        }
        
        ptoc("TPObstacleBarnesHut0::Differential");
        
    } // Differential
    
    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void TPObstacleBarnesHut0::Update()
    {
        ptic("TPObstacleBarnesHut0::Update");
        
        // Invalidate the old BVH pointer
        bvh = 0;
        // bvhSharedFrom is responsible for reallocating it in its Update() function
        bvh = bvhSharedFrom->GetBVH();
        if (!bvh)
        {
            throw std::runtime_error("Obstacle energy is sharing BVH from an energy that has no BVH.");
        }
        
        ptoc("TPObstacleBarnesHut0::Update");
    }
    
    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 TPObstacleBarnesHut0::GetExponents()
    {
        return Vector2{alpha, beta};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *TPObstacleBarnesHut0::GetBVH()
    {
        return o_bvh;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double TPObstacleBarnesHut0::GetTheta()
    {
        return theta;
    }

} // namespace rsurfaces
