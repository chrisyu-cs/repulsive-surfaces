#include "energy/tpe_all_pairs_pr.h"
#include "bct_constructors.h"

namespace rsurfaces
{

    template<typename T1, typename T2>
    mreal TPEnergyAllPairs_Projectors::Energy(T1 alphahalf, T2 betahalf)
    {
        T2 minus_betahalf = -betahalf;
        
        auto S = bvh;
        auto T = bvh;
        
        mint m = S->primitive_count;
        mint n = T->primitive_count;
        mint data_dim = S->data_dim;
        mint nthreads = std::min( S->thread_count, T->thread_count);
    
        
        // Dunno why "restrict" helps with P_data. It is actually a lie here.
        mreal const * const restrict A  = S->P_data[0];
        mreal const * const restrict X1 = S->P_data[1];
        mreal const * const restrict X2 = S->P_data[2];
        mreal const * const restrict X3 = S->P_data[3];
        mreal const * const restrict P11 = S->P_data[4];
        mreal const * const restrict P12 = S->P_data[5];
        mreal const * const restrict P13 = S->P_data[6];
        mreal const * const restrict P22 = S->P_data[7];
        mreal const * const restrict P23 = S->P_data[8];
        mreal const * const restrict P33 = S->P_data[9];
        
        mreal const * const restrict B  = T->P_data[0];
        mreal const * const restrict Y1 = T->P_data[1];
        mreal const * const restrict Y2 = T->P_data[2];
        mreal const * const restrict Y3 = T->P_data[3];
        mreal const * const restrict Q11 = T->P_data[4];
        mreal const * const restrict Q12 = T->P_data[5];
        mreal const * const restrict Q13 = T->P_data[6];
        mreal const * const restrict Q22 = T->P_data[7];
        mreal const * const restrict Q23 = T->P_data[8];
        mreal const * const restrict Q33 = T->P_data[9];
        
        mreal sum = 0.;
        #pragma omp parallel for num_threads( nthreads ) reduction( + : sum)
        for( mint i = 0; i < m ; ++i )
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
            
            // if b_i == b_j, we loop only over the upper triangular block, diagonal excluded
            // Here, one could do a bit of horizontal vectorization. However, the number of js an x interacts with varies greatly..
//            #pragma omp simd aligned( B, Y1, Y2, Y3, Q11, Q12, Q13, Q22, Q23, Q33 : ALIGN ) reduction( + : i_sum )
            for( mint j = i + 1; j < n; ++j )
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
            sum += A[i] * i_sum;
        }
        return sum;
    }; // Energy


    template<typename T1, typename T2>
    mreal TPEnergyAllPairs_Projectors::DEnergy(T1 alphahalf, T2 betahalf)
    {
        T1 alphahalf_minus_1 = alphahalf - 1;
        T2 minus_betahalf_minus_1 = -betahalf - 1;
        
        mreal beta = 2. * betahalf;
        
        mreal sum = 0.;
        
        auto S = bvh;
        auto T = bvh;
        
        mint m = S->primitive_count;
        mint n = T->primitive_count;
        mint data_dim = S->data_dim;
        mint nthreads = std::min( S->thread_count, T->thread_count);
        
        // Dunno why "restrict" helps with P_data. It is actually a lie here.
        mreal const * const restrict A  = S->P_data[0];
        mreal const * const restrict X1 = S->P_data[1];
        mreal const * const restrict X2 = S->P_data[2];
        mreal const * const restrict X3 = S->P_data[3];
        mreal const * const restrict P11 = S->P_data[4];
        mreal const * const restrict P12 = S->P_data[5];
        mreal const * const restrict P13 = S->P_data[6];
        mreal const * const restrict P22 = S->P_data[7];
        mreal const * const restrict P23 = S->P_data[8];
        mreal const * const restrict P33 = S->P_data[9];
        
        mreal const * const restrict B  = T->P_data[0];
        mreal const * const restrict Y1 = T->P_data[1];
        mreal const * const restrict Y2 = T->P_data[2];
        mreal const * const restrict Y3 = T->P_data[3];
        mreal const * const restrict Q11 = T->P_data[4];
        mreal const * const restrict Q12 = T->P_data[5];
        mreal const * const restrict Q13 = T->P_data[6];
        mreal const * const restrict Q22 = T->P_data[7];
        mreal const * const restrict Q23 = T->P_data[8];
        mreal const * const restrict Q33 = T->P_data[9];
        
        #pragma omp parallel for num_threads( nthreads ) reduction( +: sum )
        for( mint i = 0; i < m ; ++i )
        {
            
            mint thread = omp_get_thread_num();
            
            mreal * const restrict U = &S->P_D_data[thread][0];
            mreal * const restrict V = &T->P_D_data[thread][0];
            
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
            
            mreal i_sum = 0.;
        
            // Here, one could do a bit of horizontal vectorization.
            #pragma omp simd aligned( B, Y1, Y2, Y3, Q11, Q12, Q13, Q22, Q23, Q33 : ALIGN ) reduction( + : i_sum)
            for( mint j = i + 1; j < n; ++j )
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
                i_sum += a * b * E;
                
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
                
                V[ data_dim * j + 0 ] += a * ( E - dEdv1 * y1 - dEdv2 * y2 - dEdv3 * y3 - factor * rCosPsiAlpha );
                V[ data_dim * j + 1 ] += a * dEdv1;
                V[ data_dim * j + 2 ] += a * dEdv2;
                V[ data_dim * j + 3 ] += a * dEdv3;
                V[ data_dim * j + 4 ] += aG * v11;
                V[ data_dim * j + 5 ] += aG * v12;
                V[ data_dim * j + 6 ] += aG * v13;
                V[ data_dim * j + 7 ] += aG * v22;
                V[ data_dim * j + 8 ] += aG * v23;
                V[ data_dim * j + 9 ] += aG * v33;
                
            }
            
            sum += i_sum;
            
            U[ data_dim * i + 0 ] +=  da;
            U[ data_dim * i + 1 ] += dx1;
            U[ data_dim * i + 2 ] += dx2;
            U[ data_dim * i + 3 ] += dx3;
            U[ data_dim * i + 4 ] += dp11;
            U[ data_dim * i + 5 ] += dp12;
            U[ data_dim * i + 6 ] += dp13;
            U[ data_dim * i + 7 ] += dp22;
            U[ data_dim * i + 8 ] += dp23;
            U[ data_dim * i + 9 ] += dp33;
            
        }
    
        return sum;
    }; //DEnergy
    
    // Returns the current value of the energy.
    double TPEnergyAllPairs_Projectors::Value()
    {
        
        if( use_int )
        {
            mint int_alphahalf = std::round(alpha/2);
            mint int_betahalf = std::round(beta/2);
            return weight * Energy( int_alphahalf, int_betahalf );
        }
        else
        {
            mreal real_alphahalf = alpha/2;
            mreal real_betahalf = beta/2;
            return weight * Energy( real_alphahalf, real_betahalf );
        }
    } // Value

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void TPEnergyAllPairs_Projectors::Differential(Eigen::MatrixXd & output)
    {
        if( bvh->data_dim != 10)
        {
            eprint("in TPEnergyAllPairs_Projectors::Differential: data_dim != 10");
        }
        
        EigenMatrixRM P_D_data ( bvh->primitive_count , bvh->data_dim );
        
        bvh->CleanseD();
        
        if( use_int )
        {
            mint int_alphahalf = std::round(alpha/2);
            mint int_betahalf = std::round(beta/2);
            DEnergy( int_alphahalf, int_betahalf );
            
        }
        else
        {
            mreal real_alphahalf = alpha/2;
            mreal real_betahalf = beta/2;
            DEnergy( real_alphahalf, real_betahalf );
        }
        
        bvh->CollectDerivatives( P_D_data.data() );
        
        AssembleDerivativeFromACPData( mesh, geom, P_D_data, output, weight );
        
    } // Differential


    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void TPEnergyAllPairs_Projectors::Update()
    {
        if (bvh)
        {
            delete bvh;
        }

        bvh = CreateOptimizedBVH_Projectors(mesh, geom);
        
    }

    // Get the mesh associated with this energy.
    MeshPtr TPEnergyAllPairs_Projectors::GetMesh()
    {
        return mesh;
    }

    // Get the geometry associated with this geometry.
    GeomPtr TPEnergyAllPairs_Projectors::GetGeom()
    {
        return geom;
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 TPEnergyAllPairs_Projectors::GetExponents()
    {
        return Vector2{alpha, beta};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *TPEnergyAllPairs_Projectors::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double TPEnergyAllPairs_Projectors::GetTheta()
    {
        return 0.;
    }
    
    
} // namespace rsurfaces





