
#include "energy/tp_obstacle_multipole_pr_0.h"

namespace rsurfaces
{
    
    template<typename T1, typename T2>
    mreal TPObstacleMultipole_Projectors0::FarField( T1 alphahalf, T2 betahalf)
    {
        T2 minus_betahalf = -betahalf;
        
        auto S = bct->S;
        auto T = bct->T;
        mint b_m = bct->far->b_m;
        mint  const * restrict const  b_outer = bct->far->b_outer;
        mint  const * restrict const  b_inner = bct->far->b_inner;
        mint nthreads = std::min( S->thread_count, T->thread_count);
        
        // Dunno why "restrict" helps with P_near. It is actually a lie here when S = T.
        // Well, it isn't in the far field, because no cluster may interact with itself...
        mreal const * restrict const A  = S->C_far[0];
        mreal const * restrict const X1 = S->C_far[1];
        mreal const * restrict const X2 = S->C_far[2];
        mreal const * restrict const X3 = S->C_far[3];
        mreal const * restrict const P11 = S->C_far[4];
        mreal const * restrict const P12 = S->C_far[5];
        mreal const * restrict const P13 = S->C_far[6];
        mreal const * restrict const P22 = S->C_far[7];
        mreal const * restrict const P23 = S->C_far[8];
        mreal const * restrict const P33 = S->C_far[9];
        
        mreal const * restrict const B  = T->C_far[0];
        mreal const * restrict const Y1 = T->C_far[1];
        mreal const * restrict const Y2 = T->C_far[2];
        mreal const * restrict const Y3 = T->C_far[3];
        mreal const * restrict const Q11 = T->C_far[4];
        mreal const * restrict const Q12 = T->C_far[5];
        mreal const * restrict const Q13 = T->C_far[6];
        mreal const * restrict const Q22 = T->C_far[7];
        mreal const * restrict const Q23 = T->C_far[8];
        mreal const * restrict const Q33 = T->C_far[9];
        
        mreal sum = 0.;
        
        #pragma omp parallel for num_threads( nthreads ) reduction( + : sum )
        for( mint i = 0; i < b_m; ++i )
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
            
            mreal block_sum = 0.;
            
            // This loop can be SIMDized straight-forwardly (horizontal SIMDization).
            
            mint k_begin = b_outer[i];
            mint k_end = b_outer[i+1];
            #pragma omp simd aligned( B, Y1, Y2, Y3, Q11, Q12, Q13, Q22, Q23, Q33 : ALIGN )
            for( mint k = k_begin; k < k_end; ++k )
            {
                mint j = b_inner[k];

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
                
                mreal en = ( mypow( fabs(rCosPhi2), alphahalf ) + mypow( fabs(rCosPsi2), alphahalf) ) * mypow( r2, minus_betahalf );
                
                block_sum += en * B[j];
            } // for( mint k = k_begin; k < k_end; ++k )
            
            sum += A[i] * block_sum;
        } //  for( mint i = 0; i < b_m; ++i )
        return sum;
    } // FarField


    template<typename T1, typename T2>
    mreal TPObstacleMultipole_Projectors0::NearField(T1 alphahalf, T2 betahalf)
    {
        T2 minus_betahalf = -betahalf;
        
        auto S = bct->S;
        auto T = bct->T;
        mint b_m = bct->near->b_m;
        mint nthreads = std::min( S->thread_count, T->thread_count);
        
        mint  const * restrict const  b_row_ptr = S->leaf_cluster_ptr;
        mint  const * restrict const  b_col_ptr = T->leaf_cluster_ptr;
        
        mint  const * restrict const  b_outer   = bct->near->b_outer;
        mint  const * restrict const  b_inner   = bct->near->b_inner;
        
        // Dunno why "restrict" helps with P_near. It is actually a lie here.
        mreal const * restrict const A  = S->P_near[0];
        mreal const * restrict const X1 = S->P_near[1];
        mreal const * restrict const X2 = S->P_near[2];
        mreal const * restrict const X3 = S->P_near[3];
        mreal const * restrict const P11 = S->P_near[4];
        mreal const * restrict const P12 = S->P_near[5];
        mreal const * restrict const P13 = S->P_near[6];
        mreal const * restrict const P22 = S->P_near[7];
        mreal const * restrict const P23 = S->P_near[8];
        mreal const * restrict const P33 = S->P_near[9];
        
        mreal const * restrict const B  = T->P_near[0];
        mreal const * restrict const Y1 = T->P_near[1];
        mreal const * restrict const Y2 = T->P_near[2];
        mreal const * restrict const Y3 = T->P_near[3];
        mreal const * restrict const Q11 = T->P_near[4];
        mreal const * restrict const Q12 = T->P_near[5];
        mreal const * restrict const Q13 = T->P_near[6];
        mreal const * restrict const Q22 = T->P_near[7];
        mreal const * restrict const Q23 = T->P_near[8];
        mreal const * restrict const Q33 = T->P_near[9];
        
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
                    mreal p11 = P11[i];
                    mreal p12 = P12[i];
                    mreal p13 = P13[i];
                    mreal p22 = P22[i];
                    mreal p23 = P23[i];
                    mreal p33 = P33[i];
                    
                    mreal i_sum = 0.;
                    
                    // Here, one could do a bit of horizontal vectorization. However, the number of js an x interacts with varies greatly..
                    #pragma omp simd aligned( B, Y1, Y2, Y3, Q11, Q12, Q13, Q22, Q23, Q33 : ALIGN ) reduction( + : block_sum )
                    for( mint j = j_begin; j < j_end; ++j )
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
                        
                        mreal en = ( mypow( fabs(rCosPhi2), alphahalf ) + mypow( fabs(rCosPsi2), alphahalf) ) * mypow( r2, minus_betahalf );
                        
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
    mreal TPObstacleMultipole_Projectors0::DFarField(T1 alphahalf, T2 betahalf)
    {
        
        T1 alphahalf_minus_1 = alphahalf - 1;
        T2 minus_betahalf_minus_1 = -betahalf - 1;
        
        mreal beta = 2. * betahalf;
        
        mreal sum = 0.;
        
        auto S = bct->S;
        auto T = bct->T;
//        bool not_symmetric = !bct->is_symmetric;
        mint b_m = bct->far->b_m;

        mint nthreads = std::min( S->thread_count, T->thread_count);
        
        mint  const * restrict const  b_outer = bct->far->b_outer;
        mint  const * restrict const  b_inner = bct->far->b_inner;
        
        
        // Dunno why "restrict" helps with C_far. It is actually a lie here.
        mreal const * restrict const A  = S->C_far[0];
        mreal const * restrict const X1 = S->C_far[1];
        mreal const * restrict const X2 = S->C_far[2];
        mreal const * restrict const X3 = S->C_far[3];
        mreal const * restrict const P11 = S->C_far[4];
        mreal const * restrict const P12 = S->C_far[5];
        mreal const * restrict const P13 = S->C_far[6];
        mreal const * restrict const P22 = S->C_far[7];
        mreal const * restrict const P23 = S->C_far[8];
        mreal const * restrict const P33 = S->C_far[9];
        
        mreal const * restrict const B  = T->C_far[0];
        mreal const * restrict const Y1 = T->C_far[1];
        mreal const * restrict const Y2 = T->C_far[2];
        mreal const * restrict const Y3 = T->C_far[3];
        mreal const * restrict const Q11 = T->C_far[4];
        mreal const * restrict const Q12 = T->C_far[5];
        mreal const * restrict const Q13 = T->C_far[6];
        mreal const * restrict const Q22 = T->C_far[7];
        mreal const * restrict const Q23 = T->C_far[8];
        mreal const * restrict const Q33 = T->C_far[9];
        
    #pragma omp parallel for num_threads( nthreads ) reduction( + : sum )
        for( mint i = 0; i < b_m; ++i )
        {
            mint thread = omp_get_thread_num();
            
            mreal * const restrict U = &S->C_D_far[thread][0];
//            mreal * const restrict V = &T->C_D_data[thread][0];
            
            mreal a  =  A[i];
            mreal x1 = X1[i];
            mreal x2 = X2[i];
            mreal x3 = X3[i];
            mreal p11 = P11[i];
            mreal p12 = P12[i];
            mreal p13 = P13[i];
            mreal p22 = P22[i];
            mreal p23 = P23[i];
            mreal p33 = P33[i];
            
            mreal da = 0.;
            mreal dx1 = 0.;
            mreal dx2 = 0.;
            mreal dx3 = 0.;
            mreal dp11 = 0.;
            mreal dp12 = 0.;
            mreal dp13 = 0.;
            mreal dp22 = 0.;
            mreal dp23 = 0.;
            mreal dp33 = 0.;
            
            mreal block_sum = 0.;
            
            mint k_begin = b_outer[i];
            mint k_end = b_outer[i+1];
            
            // This loop can be SIMDized straight-forwardly (horizontal SIMDization).
            #pragma omp simd aligned( B, Y1, Y2, Y3, Q11, Q12, Q13, Q22, Q23, Q33 : ALIGN ) reduction( + : block_sum )
            for( mint k = k_begin; k < k_end; ++k )
            {
                mint j = b_inner[k];
                
                mreal  b = B [j];
                mreal y1 = Y1[j];
                mreal y2 = Y2[j];
                mreal y3 = Y3[j];
                mreal q11 = Q11[j];
                mreal q12 = Q12[j];
                mreal q13 = Q13[j];
                mreal q22 = Q22[j];
                mreal q23 = Q23[j];
                mreal q33 = Q33[j];
                
                
                mreal v1 = y1 - x1;
                mreal v2 = y2 - x2;
                mreal v3 = y3 - x3;
                
                mreal v11 = v1 * v1;
                mreal v22 = v2 * v2;
                mreal v33 = v3 * v3;
                
                mreal v12 = 2. * v1 * v2;
                mreal v13 = 2. * v1 * v3;
                mreal v23 = 2. * v2 * v3;
                mreal r2 = v11 + v22 + v33;
                
                mreal Pv1 = p11*v1 + p12*v2 + p13*v3;
                mreal Pv2 = p12*v1 + p22*v2 + p23*v3;
                mreal Pv3 = p13*v1 + p23*v2 + p33*v3;
                mreal rCosPhi2 = v1*Pv1 + v2*Pv2 + v3*Pv3;
                
                mreal Qv1 = q11*v1 + q12*v2 + q13*v3;
                mreal Qv2 = q12*v1 + q22*v2 + q23*v3;
                mreal Qv3 = q13*v1 + q23*v2 + q33*v3;
                mreal rCosPsi2 = v1*Qv1 + v2*Qv2 + v3*Qv3;
                
                mreal rCosPhiAlphaMinus2 = mypow( fabs(rCosPhi2), alphahalf_minus_1);
                mreal rCosPsiAlphaMinus2 = mypow( fabs(rCosPsi2), alphahalf_minus_1);
                mreal rMinusBetaMinus2 = mypow( r2, minus_betahalf_minus_1 );
                
                mreal rMinusBeta = rMinusBetaMinus2 * r2;
                mreal rCosPhiAlpha = rCosPhiAlphaMinus2 * rCosPhi2;
                mreal rCosPsiAlpha = rCosPsiAlphaMinus2 * rCosPsi2;
                mreal Num = ( rCosPhiAlpha + rCosPsiAlpha );
            
                mreal E = Num * rMinusBeta;
                block_sum += a * b * E;
                
                mreal factor = alphahalf * rMinusBeta;
                mreal F = factor * rCosPhiAlphaMinus2;
                mreal G = factor * rCosPsiAlphaMinus2;
                mreal H = - beta * rMinusBetaMinus2 * Num;
                
                mreal bF = b * F;
                mreal aG = a * G;
                
                mreal dEdv1 = 2. * (F * Pv1 + G * Qv1) + H * v1;
                mreal dEdv2 = 2. * (F * Pv2 + G * Qv2) + H * v2;
                mreal dEdv3 = 2. * (F * Pv3 + G * Qv3) + H * v3;
                
                da += b * ( E + dEdv1 * x1 + dEdv2 * x2 + dEdv3 * x3 - factor * rCosPhiAlpha );
                dx1 -= b * dEdv1;
                dx2 -= b * dEdv2;
                dx3 -= b * dEdv3;
                dp11 += bF * v11;
                dp12 += bF * v12;
                dp13 += bF * v13;
                dp22 += bF * v22;
                dp23 += bF * v23;
                dp33 += bF * v33;
                
            } // for( mint k = b_outer[i]; k < b_outer[i+1]; ++k )
            
            sum += block_sum;
            
            U[ 10 * i + 0 ] +=  da;
            U[ 10 * i + 1 ] += dx1;
            U[ 10 * i + 2 ] += dx2;
            U[ 10 * i + 3 ] += dx3;
            U[ 10 * i + 4 ] += dp11;
            U[ 10 * i + 5 ] += dp12;
            U[ 10 * i + 6 ] += dp13;
            U[ 10 * i + 7 ] += dp22;
            U[ 10 * i + 8 ] += dp23;
            U[ 10 * i + 9 ] += dp33;
        } // for( mint i = 0; i < b_m; ++i )
        
        return sum;
    }; //DFarField

    template<typename T1, typename T2>
    mreal TPObstacleMultipole_Projectors0::DNearField(T1 alphahalf, T2 betahalf)
    {
        T1 alphahalf_minus_1 = alphahalf - 1;
        T2 minus_betahalf_minus_1 = -betahalf - 1;
        
        mreal beta = 2. * betahalf;
        
        mreal sum = 0.;
        
        auto S = bct->S;
        auto T = bct->T;

        mint b_m = bct->near->b_m;

        mint nthreads = std::min( S->thread_count, T->thread_count);
        
        mint  const * restrict const  b_row_ptr = S->leaf_cluster_ptr;
        mint  const * restrict const  b_col_ptr = T->leaf_cluster_ptr;
        
        mint  const * restrict const  b_outer = &bct->near->b_outer[0];
        mint  const * restrict const  b_inner = &bct->near->b_inner[0];
        
        // Dunno why "restrict" helps with P_near. It is actually a lie here.
        mreal const * restrict const A  = S->P_near[0];
        mreal const * restrict const X1 = S->P_near[1];
        mreal const * restrict const X2 = S->P_near[2];
        mreal const * restrict const X3 = S->P_near[3];
        mreal const * restrict const P11 = S->P_near[4];
        mreal const * restrict const P12 = S->P_near[5];
        mreal const * restrict const P13 = S->P_near[6];
        mreal const * restrict const P22 = S->P_near[7];
        mreal const * restrict const P23 = S->P_near[8];
        mreal const * restrict const P33 = S->P_near[9];
        
        mreal const * restrict const B  = T->P_near[0];
        mreal const * restrict const Y1 = T->P_near[1];
        mreal const * restrict const Y2 = T->P_near[2];
        mreal const * restrict const Y3 = T->P_near[3];
        mreal const * restrict const Q11 = T->P_near[4];
        mreal const * restrict const Q12 = T->P_near[5];
        mreal const * restrict const Q13 = T->P_near[6];
        mreal const * restrict const Q22 = T->P_near[7];
        mreal const * restrict const Q23 = T->P_near[8];
        mreal const * restrict const Q33 = T->P_near[9];
        
        
        #pragma omp parallel for num_threads( nthreads ) reduction( +: sum )
        for( mint b_i = 0; b_i < b_m; ++b_i )
        {
            mint thread = omp_get_thread_num();
            
            mreal * restrict const U = &S->P_D_near[thread][0];
            
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
                    mreal p11 = P11[i];
                    mreal p12 = P12[i];
                    mreal p13 = P13[i];
                    mreal p22 = P22[i];
                    mreal p23 = P23[i];
                    mreal p33 = P33[i];
                    
                    mreal da = 0.;
                    mreal dx1 = 0.;
                    mreal dx2 = 0.;
                    mreal dx3 = 0.;
                    mreal dp11 = 0.;
                    mreal dp12 = 0.;
                    mreal dp13 = 0.;
                    mreal dp22 = 0.;
                    mreal dp23 = 0.;
                    mreal dp33 = 0.;
                    
                    // Here, one could do a bit of horizontal vectorization. However, the number of js an x interacts with is small and varies greatly..
                    #pragma omp simd aligned( B, Y1, Y2, Y3, Q11, Q12, Q13, Q22, Q23, Q33 : ALIGN ) reduction( + : sum)
                    for( mint j = j_begin; j < j_end; ++j )
                    {
                        mreal  b = B [j];
                        mreal y1 = Y1[j];
                        mreal y2 = Y2[j];
                        mreal y3 = Y3[j];
                        mreal q11 = Q11[j];
                        mreal q12 = Q12[j];
                        mreal q13 = Q13[j];
                        mreal q22 = Q22[j];
                        mreal q23 = Q23[j];
                        mreal q33 = Q33[j];
                        
                        
                        
                        mreal v1 = y1 - x1;
                        mreal v2 = y2 - x2;
                        mreal v3 = y3 - x3;
                        
                        mreal v11 = v1 * v1;
                        mreal v22 = v2 * v2;
                        mreal v33 = v3 * v3;
                        
                        mreal v12 = 2. * v1 * v2;
                        mreal v13 = 2. * v1 * v3;
                        mreal v23 = 2. * v2 * v3;
                        mreal r2 = v11 + v22 + v33;
                        
                        mreal Pv1 = p11*v1 + p12*v2 + p13*v3;
                        mreal Pv2 = p12*v1 + p22*v2 + p23*v3;
                        mreal Pv3 = p13*v1 + p23*v2 + p33*v3;
                        mreal rCosPhi2 = v1*Pv1 + v2*Pv2 + v3*Pv3;
                        
                        mreal Qv1 = q11*v1 + q12*v2 + q13*v3;
                        mreal Qv2 = q12*v1 + q22*v2 + q23*v3;
                        mreal Qv3 = q13*v1 + q23*v2 + q33*v3;
                        mreal rCosPsi2 = v1*Qv1 + v2*Qv2 + v3*Qv3;
                        
                        mreal rCosPhiAlphaMinus2 = mypow( fabs(rCosPhi2), alphahalf_minus_1);
                        mreal rCosPsiAlphaMinus2 = mypow( fabs(rCosPsi2), alphahalf_minus_1);
                        mreal rMinusBetaMinus2 = mypow( r2, minus_betahalf_minus_1 );
                        
                        mreal rMinusBeta = rMinusBetaMinus2 * r2;
                        mreal rCosPhiAlpha = rCosPhiAlphaMinus2 * rCosPhi2;
                        mreal rCosPsiAlpha = rCosPsiAlphaMinus2 * rCosPsi2;
                        mreal Num = ( rCosPhiAlpha + rCosPsiAlpha );
                    
                        mreal E = Num * rMinusBeta;
                        sum += a * b * E;
                        
                        mreal factor = alphahalf * rMinusBeta;
                        mreal F = factor * rCosPhiAlphaMinus2;
                        mreal G = factor * rCosPsiAlphaMinus2;
                        mreal H = - beta * rMinusBetaMinus2 * Num;
                        
                        mreal bF = b * F;
                        mreal aG = a * G;
                        
                        mreal dEdv1 = 2. * (F * Pv1 + G * Qv1) + H * v1;
                        mreal dEdv2 = 2. * (F * Pv2 + G * Qv2) + H * v2;
                        mreal dEdv3 = 2. * (F * Pv3 + G * Qv3) + H * v3;
                        
                        da += b * ( E + dEdv1 * x1 + dEdv2 * x2 + dEdv3 * x3 - factor * rCosPhiAlpha );
                        dx1 -= b * dEdv1;
                        dx2 -= b * dEdv2;
                        dx3 -= b * dEdv3;
                        dp11 += bF * v11;
                        dp12 += bF * v12;
                        dp13 += bF * v13;
                        dp22 += bF * v22;
                        dp23 += bF * v23;
                        dp33 += bF * v33;
                        
                    } // for( mint j = j_begin; j < j_end; ++j )
                    
                    U[ 10 * i + 0 ] +=  da;
                    U[ 10 * i + 1 ] += dx1;
                    U[ 10 * i + 2 ] += dx2;
                    U[ 10 * i + 3 ] += dx3;
                    U[ 10 * i + 4 ] += dp11;
                    U[ 10 * i + 5 ] += dp12;
                    U[ 10 * i + 6 ] += dp13;
                    U[ 10 * i + 7 ] += dp22;
                    U[ 10 * i + 8 ] += dp23;
                    U[ 10 * i + 9 ] += dp33;
                    
                }// for( mint i = i_begin; i < i_end ; ++i )
            }// for( mint k = b_outer[b_i]; k < b_outer[b_i+1]; ++k )
        } // for( mint b_i = 0; b_i < b_m; ++b_i )
        
        return sum;
    }; //DNearField

    // Returns the current value of the energy.
    double TPObstacleMultipole_Projectors0::Value()
    {
        
        if( use_int )
        {
            mint int_alphahalf = std::round(alpha/2);
            mint int_betahalf = std::round(beta/2);
            return weight * (FarField( int_alphahalf, int_betahalf ) + NearField (int_alphahalf, int_betahalf ));
        }
        else
        {
            mreal real_alphahalf = alpha/2;
            mreal real_betahalf = beta/2;
            return weight * (FarField( real_alphahalf, real_betahalf ) + NearField( real_alphahalf, real_betahalf ));
        }
    } // Value

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void TPObstacleMultipole_Projectors0::Differential(Eigen::MatrixXd &output)
    {
        if( bct->S->near_dim != 10)
        {
            eprint("in TPEnergyBarnesHut_Projectors0::Differential: S->near_dim != 10");
        }
        if( bct->T->near_dim != 10)
        {
            eprint("in TPEnergyBarnesHut_Projectors0::Differential: T->near_dim != 10");
        }
        if( bct->S->far_dim != 10)
        {
            eprint("in TPEnergyBarnesHut_Projectors0::Differential: S->far_dim != 10");
        }
        if( bct->T->far_dim != 10)
        {
            eprint("in TPEnergyBarnesHut_Projectors0::Differential: T->far_dim != 10");
        }
        
        bct->S->CleanseD();
//        bct->T->CleanseD();
        
        if( use_int )
        {
            mint int_alphahalf = std::round(alpha/2);
            mint int_betahalf = std::round(beta/2);
            DNearField( int_alphahalf, int_betahalf );
            DFarField ( int_alphahalf, int_betahalf );
            
        }
        else
        {
            mreal real_alphahalf = alpha/2;
            mreal real_betahalf = beta/2;
            DNearField( real_alphahalf, real_betahalf );
            DFarField ( real_alphahalf, real_betahalf );
        }
        
        EigenMatrixRM P_D_near_( bct->S->primitive_count , bct->S->near_dim );
        EigenMatrixRM P_D_far_ ( bct->S->primitive_count , bct->S->far_dim );
        
        bct->S->CollectDerivatives( P_D_near_.data(), P_D_far_.data() );
        
        AssembleDerivativeFromACPData( mesh, geom, P_D_near_, output, weight );
        AssembleDerivativeFromACPData( mesh, geom, P_D_far_, output, weight );
        
    } // Differential


    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void TPObstacleMultipole_Projectors0::Update()
    {
        // Nothing needs to be done
    }

    // Get the mesh associated with this energy.
    MeshPtr TPObstacleMultipole_Projectors0::GetMesh()
    {
        return mesh;
    }

    // Get the geometry associated with this geometry.
    GeomPtr TPObstacleMultipole_Projectors0::GetGeom()
    {
        return geom;
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 TPObstacleMultipole_Projectors0::GetExponents()
    {
        return Vector2{alpha, beta};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *TPObstacleMultipole_Projectors0::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double TPObstacleMultipole_Projectors0::GetTheta()
    {
        return sqrt(bct->theta2);
    }

} // namespace rsurfaces
