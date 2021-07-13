
#include "energy/tp_obstacle_multipole_0.h"

namespace rsurfaces
{

    template<typename T1, typename T2>
    mreal TPObstacleCutOff::NearField(T1 alpha, T2 betahalf)
    {
        T2 minus_betahalf = -betahalf;
        
        mreal squared_inv_cut_off = bct->squared_inv_cut_off;
        
        auto S = bct->S;
        auto T = bct->T;
        mint b_m = bct->near->b_m;
        mint nthreads = std::min( S->thread_count, T->thread_count);
        
        mint  const * restrict const b_row_ptr = S->leaf_cluster_ptr;
        mint  const * restrict const b_col_ptr = T->leaf_cluster_ptr;
        
        mint  const * restrict const b_outer   = bct->near->b_outer;
        mint  const * restrict const b_inner   = bct->near->b_inner;
        
        // Dunno why "restrict" helps with P_near. It is actually a lie here.
        mreal const * restrict const A  = S->P_near[0];
        mreal const * restrict const X1 = S->P_near[1];
        mreal const * restrict const X2 = S->P_near[2];
        mreal const * restrict const X3 = S->P_near[3];
        mreal const * restrict const N1 = S->P_near[4];
        mreal const * restrict const N2 = S->P_near[5];
        mreal const * restrict const N3 = S->P_near[6];
        
        mreal const * restrict const B  = T->P_near[0];
        mreal const * restrict const Y1 = T->P_near[1];
        mreal const * restrict const Y2 = T->P_near[2];
        mreal const * restrict const Y3 = T->P_near[3];
        mreal const * restrict const M1 = T->P_near[4];
        mreal const * restrict const M2 = T->P_near[5];
        mreal const * restrict const M3 = T->P_near[6];
        
        mreal sum = 0.;
        #pragma omp parallel for num_threads( nthreads ) reduction( + : sum)
        for( mint b_i = 0; b_i < b_m; ++b_i )
        {
            
            mint i_begin = b_row_ptr[b_i];
            mint i_end   = b_row_ptr[b_i+1];
            
            for( mint k = b_outer[b_i]; k < b_outer[b_i+1]; ++k )
            {
                mint b_j = b_inner[k];
                mint j_begin = b_col_ptr[b_j];
                mint j_end   = b_col_ptr[b_j+1];
                mreal block_sum = 0.;
                
                for( mint i = i_begin; i < i_end ; ++i )
                {
                    mreal x1 = X1[i];
                    mreal x2 = X2[i];
                    mreal x3 = X3[i];
                    mreal n1 = N1[i];
                    mreal n2 = N2[i];
                    mreal n3 = N3[i];
                    
                    mreal i_sum = 0.;
                    
                    // Here, one could do a bit of horizontal vectorization. However, the number of js an x interacts with varies greatly..
                    #pragma omp simd aligned( B, Y1, Y2, Y3, M1, M2, M3 : ALIGN ) reduction( + : block_sum )
                    for( mint j = j_begin; j < j_end; ++j )
                    {
                        mreal v1 = Y1[j] - x1;
                        mreal v2 = Y2[j] - x2;
                        mreal v3 = Y3[j] - x3;
                        mreal m1 = M1[j];
                        mreal m2 = M2[j];
                        mreal m3 = M3[j];
                        
                        mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                        mreal rCosPsi = v1 * m1 + v2 * m2 + v3 * m3;
                        mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 ;
                        
                        mreal mollifier = (1.- r2 * squared_inv_cut_off);
                        mollifier = mollifier * mollifier;
                        
                        mreal en = mollifier * ( mypow( fabs(rCosPhi), alpha ) + mypow( fabs(rCosPsi), alpha) ) * mypow( r2, minus_betahalf );
                        
                        
                        i_sum += en * B[j];
                    }
                    block_sum += A[i] * i_sum;
                }
                sum += block_sum;
            }
        }
        return sum;
    }; //NearField

    template<typename T1, typename T2>
    mreal TPObstacleCutOff::DNearField(T1 alpha, T2 betahalf)
    {
        T1 alpha_minus_2 = alpha - 2;
        T2 minus_betahalf_minus_1 = -betahalf - 1;
        
        mreal beta = 2. * betahalf;
        
        mreal sum = 0.;
        
        mreal squared_inv_cut_off = bct->squared_inv_cut_off;
        
        auto S = bct->S;
        auto T = bct->T;
        
        mint b_m = bct->near->b_m;
        
        mint nthreads = std::min( S->thread_count, T->thread_count);
        
        mint  const * restrict const b_row_ptr = S->leaf_cluster_ptr;
        mint  const * restrict const b_col_ptr = T->leaf_cluster_ptr;
        
        mint  const * restrict const b_outer = &bct->near->b_outer[0];
        mint  const * restrict const b_inner = &bct->near->b_inner[0];
        
        // Dunno why "restrict" helps with P_near. It is actually a lie here.
        mreal const * restrict const A  = S->P_near[0];
        mreal const * restrict const X1 = S->P_near[1];
        mreal const * restrict const X2 = S->P_near[2];
        mreal const * restrict const X3 = S->P_near[3];
        mreal const * restrict const N1 = S->P_near[4];
        mreal const * restrict const N2 = S->P_near[5];
        mreal const * restrict const N3 = S->P_near[6];
        
        mreal const * restrict const B  = T->P_near[0];
        mreal const * restrict const Y1 = T->P_near[1];
        mreal const * restrict const Y2 = T->P_near[2];
        mreal const * restrict const Y3 = T->P_near[3];
        mreal const * restrict const M1 = T->P_near[4];
        mreal const * restrict const M2 = T->P_near[5];
        mreal const * restrict const M3 = T->P_near[6];
        
        
        #pragma omp parallel for num_threads( nthreads ) reduction( +: sum )
        for( mint b_i = 0; b_i < b_m; ++b_i )
        {
            mint thread = omp_get_thread_num();
            
            mreal * restrict const U = &S->P_D_near[thread][0];
            //            mreal * restrict const V = &T->P_D_data[thread][0];
            
            mint i_begin = b_row_ptr[b_i];
            mint i_end   = b_row_ptr[b_i+1];
            
            for( mint k = b_outer[b_i]; k < b_outer[b_i+1]; ++k )
            {
                mint b_j = b_inner[k];
                mint j_begin = b_col_ptr[b_j];
                mint j_end   = b_col_ptr[b_j+1];
                
                for( mint i = i_begin; i < i_end ; ++i )
                {
                    mreal  a = A [i];
                    mreal x1 = X1[i];
                    mreal x2 = X2[i];
                    mreal x3 = X3[i];
                    mreal n1 = N1[i];
                    mreal n2 = N2[i];
                    mreal n3 = N3[i];
                    
                    mreal  da = 0.;
                    mreal dx1 = 0.;
                    mreal dx2 = 0.;
                    mreal dx3 = 0.;
                    mreal dn1 = 0.;
                    mreal dn2 = 0.;
                    mreal dn3 = 0.;
                    
                    // Here, one could do a bit of horizontal vectorization. However, the number of js an x interacts with is small and varies greatly..
                    #pragma omp simd aligned( B, Y1, Y2, Y3, M1, M2, M3 : ALIGN ) reduction( + : sum)
                    for( mint j = j_begin; j < j_end; ++j )
                    {
                        mreal  b = B [j];
                        mreal y1 = Y1[j];
                        mreal y2 = Y2[j];
                        mreal y3 = Y3[j];
                        mreal m1 = M1[j];
                        mreal m2 = M2[j];
                        mreal m3 = M3[j];
                        
                        mreal v1 = y1 - x1;
                        mreal v2 = y2 - x2;
                        mreal v3 = y3 - x3;
                        
                        mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                        mreal rCosPsi = v1 * m1 + v2 * m2 + v3 * m3;
                        mreal r2      = v1 * v1 + v2 * v2 + v3 * v3;
                        
                        mreal mollifier = (1.- r2 * squared_inv_cut_off);
                        mollifier = mollifier * mollifier;
                        
                        mreal rBetaMinus2 = mypow( r2, minus_betahalf_minus_1 );
                        mreal rBeta = rBetaMinus2 * r2;
                        
                        mreal rCosPhiAlphaMinus1 = mypow( fabs(rCosPhi), alpha_minus_2 ) * rCosPhi;
                        mreal rCosPhiAlpha = rCosPhiAlphaMinus1 * rCosPhi;
                        
                        mreal rCosPsiAlphaMinus1 = mypow( fabs(rCosPsi), alpha_minus_2 ) * rCosPsi;
                        mreal rCosPsiAlpha = rCosPsiAlphaMinus1 * rCosPsi;
                        
                        
                        mreal Num = rCosPhiAlpha + rCosPsiAlpha;
                        mreal factor0 = rBeta * alpha;
                        mreal density = rBeta * Num;
                        sum += a * b * density;
                        
                        mreal F = factor0 * rCosPhiAlphaMinus1;
                        mreal G = factor0 * rCosPsiAlphaMinus1;
                        mreal H = beta * rBetaMinus2 * Num;
                        
                        mreal bF = b * F;
                        mreal aG = a * G;
                        
                        mreal Z1 = ( - n1 * F - m1 * G + v1 * H );
                        mreal Z2 = ( - n2 * F - m2 * G + v2 * H );
                        mreal Z3 = ( - n3 * F - m3 * G + v3 * H );
                        
                        da += b * (
                                   density
                                   +
                                   F * ( n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3) )
                                   +
                                   G * ( m1 * x1 + m2 * x2 + m3 * x3 )
                                   -
                                   H * ( v1 * x1 + v2 * x2 + v3 * x3 )
                                   );
                        
                        dx1 += b  * Z1;
                        dx2 += b  * Z2;
                        dx3 += b  * Z3;
                        dn1 += bF * v1;
                        dn2 += bF * v2;
                        dn3 += bF * v3;
        
                    } // for( mint j = j_begin; j < j_end; ++j )
                    
                    U[ 7 * i     ] +=  da;
                    U[ 7 * i + 1 ] += dx1;
                    U[ 7 * i + 2 ] += dx2;
                    U[ 7 * i + 3 ] += dx3;
                    U[ 7 * i + 4 ] += dn1;
                    U[ 7 * i + 5 ] += dn2;
                    U[ 7 * i + 6 ] += dn3;
                    
                }// for( mint i = i_begin; i < i_end ; ++i )
            }// for( mint k = b_outer[b_i]; k < b_outer[b_i+1]; ++k )
        } // for( mint b_i = 0; b_i < b_m; ++b_i )
        
        return sum;
    }; //DNearField
        
    // Returns the current value of the energy.
    double TPObstacleCutOff::Value()
    {
        mreal value = 0.;
        
        mreal intpart;
        
        bool betahalfint = (std::modf( beta/2, &intpart) == 0.0);
        bool alphahalfint = (std::modf( alpha/2, &intpart) == 0.0);
        bool alphaint = (std::modf( alpha, &intpart) == 0.0);
        
        if( use_int && betahalfint && alphaint)
        {
            mint int_alpha = std::round(alpha);
            mint int_betahalf = std::round(beta/2);
            value += NearField(int_alpha, int_betahalf );
        }
        else
        {
            mreal real_alpha = alpha;
            mreal real_betahalf = beta/2;
            value += NearField( real_alpha, real_betahalf );
        }
        
        return weight * value;
    } // Value

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void TPObstacleCutOff::Differential(Eigen::MatrixXd &output)
    {
        if( bct->S->near_dim != 7)
        {
            eprint("in TPEnergyBarnesHut_Projectors0::Differential: near_dim != 7");
        }
        
        
        if( bct->T->near_dim != 7)
        {
            eprint("in TPEnergyBarnesHut_Projectors0::Differential: near_dim != 7");
        }
        
        bct->S->CleanseD();
//        bct->T->CleanseD();
        
        mreal intpart;
        
        bool betahalfint = (std::modf( beta/2, &intpart) == 0.0);
        bool alphahalfint = (std::modf( alpha/2, &intpart) == 0.0);
        bool alphaint = (std::modf( alpha, &intpart) == 0.0);
        
        if( use_int && betahalfint && alphaint)
        {
            mint int_alpha = std::round(alpha);
            mint int_betahalf = std::round(beta/2);
            DNearField( int_alpha, int_betahalf );
        }
        else
        {
            mreal real_alpha = alpha;
            mreal real_betahalf = beta/2;
            DNearField( real_alpha, real_betahalf );
        }
        
        EigenMatrixRM P_D_near( bct->S->primitive_count, bct->S->near_dim );
        
        bct->S->CollectDerivatives( P_D_near.data() );
        
        AssembleDerivativeFromACNData( mesh, geom, P_D_near, output, weight );
        
    } // Differential

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 TPObstacleCutOff::GetExponents()
    {
        return Vector2{bct->alpha, bct->beta};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *TPObstacleCutOff::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double TPObstacleCutOff::GetTheta()
    {
        return 0.;
    }

} // namespace rsurfaces
