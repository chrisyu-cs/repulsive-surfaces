#include "energy/tpe_all_pairs.h"
#include "bct_constructors.h"

namespace rsurfaces
{

    template<typename T1, typename T2>
    mreal TPEnergyAllPairs::Energy(T1 alpha, T2 betahalf)
    {
        ptic("TPEnergyAllPairs::Energy");
        
        T2 minus_betahalf = -betahalf;
        
        auto S = bvh;
        auto T = bvh;
        mint n = bvh->primitive_count;
        mint nthreads = bvh->thread_count;
        
        // Dunno why "restrict" helps with P_near. It is actually a lie here.
        mreal const * restrict const A  = bvh->P_near[0];
        mreal const * restrict const X1 = bvh->P_near[1];
        mreal const * restrict const X2 = bvh->P_near[2];
        mreal const * restrict const X3 = bvh->P_near[3];
        mreal const * restrict const N1 = bvh->P_near[4];
        mreal const * restrict const N2 = bvh->P_near[5];
        mreal const * restrict const N3 = bvh->P_near[6];
        
        mreal const * restrict const B  = bvh->P_near[0];
        mreal const * restrict const Y1 = bvh->P_near[1];
        mreal const * restrict const Y2 = bvh->P_near[2];
        mreal const * restrict const Y3 = bvh->P_near[3];
        mreal const * restrict const M1 = bvh->P_near[4];
        mreal const * restrict const M2 = bvh->P_near[5];
        mreal const * restrict const M3 = bvh->P_near[6];
        
        mreal sum = 0.;
        #pragma omp parallel for num_threads( nthreads ) reduction( + : sum)
        for( mint i = 0; i < n ; ++i )
        {
            mreal x1 = X1[i];
            mreal x2 = X2[i];
            mreal x3 = X3[i];
            mreal n1 = N1[i];
            mreal n2 = N2[i];
            mreal n3 = N3[i];
            
            mreal i_sum = 0.;
            
            // if b_i == b_j, we loop only over the upper triangular block, diagonal excluded
            // Here, one could do a bit of horizontal vectorization. However, the number of js an x interacts with varies greatly..
            #pragma omp simd aligned( B, Y1, Y2, Y3, M1, M2, M3 : ALIGN ) reduction( + : i_sum )
            for( mint j = i + 1; j < n; ++j )
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
                
                mreal en = ( mypow( fabs(rCosPhi), alpha ) + mypow( fabs(rCosPsi), alpha) ) * mypow( r2, minus_betahalf );
                
                
                i_sum += en * B[j];
            }
            sum += A[i] * i_sum;
        }
        ptoc("TPEnergyAllPairs::Energy");
        return sum;
    }; // Energy


    template<typename T1, typename T2>
    mreal TPEnergyAllPairs::DEnergy(T1 alpha, T2 betahalf)
    {
        ptic("TPEnergyAllPairs::DEnergy");
        T1 alpha_minus_2 = alpha - 2;
        T2 minus_betahalf_minus_1 = -betahalf - 1;
        
        mreal beta = 2. * betahalf;
        
        mreal sum = 0.;

        mint n = bvh->primitive_count;
        
        mint nthreads = bvh->thread_count;
        
        // Dunno why "restrict" helps with P_near. It is actually a lie here.
        mreal const * restrict const A  = bvh->P_near[0];
        mreal const * restrict const X1 = bvh->P_near[1];
        mreal const * restrict const X2 = bvh->P_near[2];
        mreal const * restrict const X3 = bvh->P_near[3];
        mreal const * restrict const N1 = bvh->P_near[4];
        mreal const * restrict const N2 = bvh->P_near[5];
        mreal const * restrict const N3 = bvh->P_near[6];
        
        mreal const * restrict const B  = bvh->P_near[0];
        mreal const * restrict const Y1 = bvh->P_near[1];
        mreal const * restrict const Y2 = bvh->P_near[2];
        mreal const * restrict const Y3 = bvh->P_near[3];
        mreal const * restrict const M1 = bvh->P_near[4];
        mreal const * restrict const M2 = bvh->P_near[5];
        mreal const * restrict const M3 = bvh->P_near[6];
        
        #pragma omp parallel for num_threads( nthreads ) reduction( +: sum )
        for( mint i = 0; i < n ; ++i )
        {
            mint thread = omp_get_thread_num();
            
            mreal * restrict const U = &bvh->P_D_near[thread][0];
            mreal * restrict const V = &bvh->P_D_near[thread][0];
            
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
            
            mreal i_sum = 0.;
        
            // Here, one could do a bit of horizontal vectorization.
            #pragma omp simd aligned( B, Y1, Y2, Y3, M1, M2, M3 : ALIGN ) reduction( + : i_sum)
            for( mint j = i + 1; j < n; ++j )
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
                
                mreal rBetaMinus2 = mypow( r2, minus_betahalf_minus_1 );
                mreal rBeta = rBetaMinus2 * r2;
                
                mreal rCosPhiAlphaMinus1 = mypow( fabs(rCosPhi), alpha_minus_2 ) * rCosPhi;
                mreal rCosPhiAlpha = rCosPhiAlphaMinus1 * rCosPhi;
                
                mreal rCosPsiAlphaMinus1 = mypow( fabs(rCosPsi), alpha_minus_2 ) * rCosPsi;
                mreal rCosPsiAlpha = rCosPsiAlphaMinus1 * rCosPsi;
                
                
                mreal Num = rCosPhiAlpha + rCosPsiAlpha;
                mreal factor0 = rBeta * alpha;
                mreal density = rBeta * Num;
                i_sum += a * b * density;
                
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
                
                V[ 7 * j ] += a * (
                                   density
                                   -
                                   F * ( n1 * y1 + n2 * y2 + n3 * y3 )
                                   -
                                   G * ( m1 * (y1 + v1) + m2 * (y2 + v2) + m3 * (y3 + v3) )
                                   +
                                   H * ( v1 * y1 + v2 * y2 + v3 * y3 )
                                   );
                
                dx1 += b  * Z1;
                dx2 += b  * Z2;
                dx3 += b  * Z3;
                dn1 += bF * v1;
                dn2 += bF * v2;
                dn3 += bF * v3;
                
                V[ 7 * j + 1 ] -= a  * Z1;
                V[ 7 * j + 2 ] -= a  * Z2;
                V[ 7 * j + 3 ] -= a  * Z3;
                V[ 7 * j + 4 ] += aG * v1;
                V[ 7 * j + 5 ] += aG * v2;
                V[ 7 * j + 6 ] += aG * v3;
            }// for( mint j = begin; j < T_n; ++j )
            
            sum += i_sum;
            
            U[ 7 * i     ] +=  da;
            U[ 7 * i + 1 ] += dx1;
            U[ 7 * i + 2 ] += dx2;
            U[ 7 * i + 3 ] += dx3;
            U[ 7 * i + 4 ] += dn1;
            U[ 7 * i + 5 ] += dn2;
            U[ 7 * i + 6 ] += dn3;
            
        }// for( mint i = 0; i < S_n ; ++i )
        ptoc("TPEnergyAllPairs::DEnergy");
        return sum;
    }; //DEnergy
    
    // Returns the current value of the energy.
    double TPEnergyAllPairs::Value()
    {
        ptic("TPEnergyAllPairs::Value");
        
        mreal value = 0.;
        
        if( use_int )
        {
            mint int_alpha = std::round(alpha);
            mint int_betahalf = std::round(beta/2);
            value = weight * Energy( int_alpha, int_betahalf );
        }
        else
        {
            mreal real_alpha = alpha;
            mreal real_betahalf = beta/2;
            value = weight * Energy( real_alpha, real_betahalf );
        }
        
        ptoc("TPEnergyAllPairs::Value");
        
        return value;
    } // Value

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void TPEnergyAllPairs::Differential(Eigen::MatrixXd &output)
    {
        ptic("TPEnergyAllPairs::Differential");
        
        if( bvh->near_dim != 7)
        {
            eprint("in TPEnergyAllPairs::Differential: near_dim != 7");
        }
        
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
        
        EigenMatrixRM P_D_near( bvh->primitive_count, bvh->near_dim );
        
        bvh->CollectDerivatives( P_D_near.data() );
    
        AssembleDerivativeFromACNData( mesh, geom, P_D_near, output, weight );
        
        ptoc("TPEnergyAllPairs::Differential");
        
    } // Differential


    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void TPEnergyAllPairs::Update()
    {
        ptic("TPEnergyAllPairs::Update");
        if (bvh)
        {
            delete bvh;
        }
        
        bvh = CreateOptimizedBVH(mesh, geom);
        
        ptoc("TPEnergyAllPairs::Update");
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 TPEnergyAllPairs::GetExponents()
    {
        return Vector2{alpha, beta};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *TPEnergyAllPairs::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double TPEnergyAllPairs::GetTheta()
    {
        return 0.;
    }

} // namespace rsurfaces
