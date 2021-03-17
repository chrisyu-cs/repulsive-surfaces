#include "energy/tpe_all_pairs.h"
#include "bct_constructors.h"

namespace rsurfaces
{

    // Returns the current value of the energy.
    double TPEnergyAllPairs::Value()
    {
        
        if( use_int )
        {
            mint int_alpha = std::round(alpha);
            mint int_betahalf = std::round(beta/2);
            return Energy( int_alpha, int_betahalf );
        }
        else
        {
            mreal real_alpha = alpha;
            mreal real_betahalf = beta/2;
            return Energy( real_alpha, real_betahalf );
        }
    } // Value

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void TPEnergyAllPairs::Differential(Eigen::MatrixXd &output)
    {
        EigenMatrixRM P_D_data ( mesh->nFaces(), 7 );
        
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
                
        mint vertex_count = mesh->nVertices();
        VertexIndices vInds = mesh->getVertexIndices();
        FaceIndices fInds = mesh->getFaceIndices();
        
        geom->requireVertexDualAreas();
        geom->requireFaceAreas();
        geom->requireCotanLaplacian();
        geom->requireVertexPositions();
        
        Eigen::MatrixXd buffer ( mesh->nFaces() * 3, 3 );
        
        for( auto face : mesh->faces() )
        {
            mint i = fInds[face];
            
            GCHalfedge he = face.halfedge();
            
            mint i0 = vInds[he.vertex()];
            mint i1 = vInds[he.next().vertex()];
            mint i2 = vInds[he.next().next().vertex()];
            
            mreal x00 = geom->inputVertexPositions[i0][0];
            mreal x01 = geom->inputVertexPositions[i0][1];
            mreal x02 = geom->inputVertexPositions[i0][2];
            
            mreal x10 = geom->inputVertexPositions[i1][0];
            mreal x11 = geom->inputVertexPositions[i1][1];
            mreal x12 = geom->inputVertexPositions[i1][2];
            
            mreal x20 = geom->inputVertexPositions[i2][0];
            mreal x21 = geom->inputVertexPositions[i2][1];
            mreal x22 = geom->inputVertexPositions[i2][2];
            
            mreal s0 = x11;
            mreal s1 = x20;
            mreal s2 = x10;
            mreal s3 = x21;
            mreal s4 = -s0;
            mreal s5 = s3 + s4;
            mreal s6 = -s1;
            mreal s7 = s2 + s6;
            mreal s8 = x12;
            mreal s9 = x22;
            mreal s10 = x00;
            mreal s11 = -s8;
            mreal s12 = s11 + s9;
            mreal s13 = x01;
            mreal s14 = s13*s7;
            mreal s15 = s0*s1;
            mreal s16 = -(s2*s3);
            mreal s17 = s10*s5;
            mreal s18 = s14 + s15 + s16 + s17;
            mreal s19 = s18*s18;
            mreal s20 = x02;
            mreal s21 = s20*s7;
            mreal s22 = s1*s8;
            mreal s23 = -(s2*s9);
            mreal s24 = s10*s12;
            mreal s25 = s21 + s22 + s23 + s24;
            mreal s26 = s25*s25;
            mreal s27 = -s3;
            mreal s28 = s0 + s27;
            mreal s29 = s20*s28;
            mreal s30 = s3*s8;
            mreal s31 = -(s0*s9);
            mreal s32 = s12*s13;
            mreal s33 = s29 + s30 + s31 + s32;
            mreal s34 = s33*s33;
            mreal s35 = s19 + s26 + s34;
            mreal s36 = sqrt(s35);
            mreal s37 = 1/s36;
            mreal s38 = 2*s18*s5;
            mreal s39 = 2*s12*s25;
            mreal s40 = s38 + s39;
            mreal s41 = P_D_data( i , 0 );
            mreal s42 = s1 + s10 + s2;
            mreal s43 = 2*s18*s7;
            mreal s44 = 2*s12*s33;
            mreal s45 = s43 + s44;
            mreal s46 = P_D_data( i , 1 );
            mreal s47 = s0 + s13 + s3;
            mreal s48 = s36/6.;
            mreal s49 = P_D_data( i , 2 );
            mreal s50 = s20 + s8 + s9;
            mreal s51 = P_D_data( i , 3 );
            mreal s52 = P_D_data( i , 6 );
            mreal s53 = 2*s25*s7;
            mreal s54 = 2*s28*s33;
            mreal s55 = s53 + s54;
            mreal s56 = P_D_data( i , 4 );
            mreal s57 = P_D_data( i , 5 );
            mreal s58 = -s9;
            mreal s59 = s13 + s27;
            mreal s60 = 2*s18*s59;
            mreal s61 = s20 + s58;
            mreal s62 = 2*s25*s61;
            mreal s63 = s60 + s62;
            mreal s64 = -s10;
            mreal s65 = s1 + s64;
            mreal s66 = 2*s18*s65;
            mreal s67 = 2*s33*s61;
            mreal s68 = s66 + s67;
            mreal s69 = -s13;
            mreal s70 = s3 + s69;
            mreal s71 = 2*s25*s65;
            mreal s72 = 2*s33*s70;
            mreal s73 = s71 + s72;
            mreal s74 = -s20;
            mreal s75 = s0 + s69;
            mreal s76 = 2*s18*s75;
            mreal s77 = s74 + s8;
            mreal s78 = 2*s25*s77;
            mreal s79 = s76 + s78;
            mreal s80 = -s2;
            mreal s81 = s10 + s80;
            mreal s82 = 2*s18*s81;
            mreal s83 = 2*s33*s77;
            mreal s84 = s82 + s83;
            mreal s85 = s13 + s4;
            mreal s86 = 2*s25*s81;
            mreal s87 = 2*s33*s85;
            mreal s88 = s86 + s87;
            buffer( 3 * i + 0, 0 ) = (s37*s40*s41)/4. + s46*((s37*s40*s42)/12. + s48) + (s37*s40*s47*s49)/12. + (s37*s40*s50*s51)/12. + (s28*s52)/2. + (s12*s57)/2.;
            buffer( 3 * i + 0, 1 ) = (s37*s41*s45)/4. + (s37*s42*s45*s46)/12. + ((s37*s45*s47)/12. + s48)*s49 + (s37*s45*s50*s51)/12. + (s56*(s58 + s8))/2. + (s52*(s1 + s80))/2.;
            buffer( 3 * i + 0, 2 ) = (s37*s41*s55)/4. + (s37*s42*s46*s55)/12. + (s37*s47*s49*s55)/12. + s51*(s48 + (s37*s50*s55)/12.) + (s5*s56)/2. + (s57*s7)/2.;
            buffer( 3 * i + 1, 0 ) = (s57*s61)/2. + (s37*s41*s63)/4. + (s37*s47*s49*s63)/12. + (s37*s50*s51*s63)/12. + s46*(s48 + (s37*s42*s63)/12.) + (s52*s70)/2.;
            buffer( 3 * i + 1, 1 ) = (s52*(s10 + s6))/2. + (s37*s41*s68)/4. + (s37*s42*s46*s68)/12. + (s37*s50*s51*s68)/12. + s49*(s48 + (s37*s47*s68)/12.) + (s56*(s74 + s9))/2.;
            buffer( 3 * i + 1, 2 ) = (s56*s59)/2. + (s57*s65)/2. + (s37*s41*s73)/4. + (s37*s42*s46*s73)/12. + (s37*s47*s49*s73)/12. + s51*(s48 + (s37*s50*s73)/12.);
            buffer( 3 * i + 2, 0 ) = (s57*s77)/2. + (s37*s41*s79)/4. + (s37*s47*s49*s79)/12. + (s37*s50*s51*s79)/12. + s46*(s48 + (s37*s42*s79)/12.) + (s52*s85)/2.;
            buffer( 3 * i + 2, 1 ) = ((s11 + s20)*s56)/2. + (s52*(s2 + s64))/2. + (s37*s41*s84)/4. + (s37*s42*s46*s84)/12. + (s37*s50*s51*s84)/12. + s49*(s48 + (s37*s47*s84)/12.);
            buffer( 3 * i + 2, 2 ) = (s56*s75)/2. + (s57*s81)/2. + (s37*s41*s88)/4. + (s37*s42*s46*s88)/12. + (s37*s47*s49*s88)/12. + s51*(s48 + (s37*s50*s88)/12.);
        }
        
        output = DerivativeAssembler( mesh, geom ) * buffer;
        
    } // Differential


    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void TPEnergyAllPairs::Update()
    {
        if (bvh)
        {
            delete bvh;
        }
        
        bvh = CreateOptimizedBVH(mesh, geom);
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


    template<typename T1, typename T2>
    mreal TPEnergyAllPairs::Energy(T1 alpha, T2 betahalf)
    {
        T2 minus_betahalf = -betahalf;
        
        auto S = bvh;
        auto T = bvh;
        mint n = bvh->primitive_count;
        mint nthreads = bvh->thread_count;
        
        // Dunno why "restrict" helps with P_data. It is actually a lie here.
        mreal const * const restrict A  = bvh->P_data[0];
        mreal const * const restrict X1 = bvh->P_data[1];
        mreal const * const restrict X2 = bvh->P_data[2];
        mreal const * const restrict X3 = bvh->P_data[3];
        mreal const * const restrict N1 = bvh->P_data[4];
        mreal const * const restrict N2 = bvh->P_data[5];
        mreal const * const restrict N3 = bvh->P_data[6];
        
        mreal const * const restrict B  = bvh->P_data[0];
        mreal const * const restrict Y1 = bvh->P_data[1];
        mreal const * const restrict Y2 = bvh->P_data[2];
        mreal const * const restrict Y3 = bvh->P_data[3];
        mreal const * const restrict M1 = bvh->P_data[4];
        mreal const * const restrict M2 = bvh->P_data[5];
        mreal const * const restrict M3 = bvh->P_data[6];
        
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
        return sum;
    }; // Energy


    template<typename T1, typename T2>
    mreal TPEnergyAllPairs::DEnergy(T1 alpha, T2 betahalf)
    {
        T1 alpha_minus_2 = alpha - 2;
        T2 minus_betahalf_minus_1 = -betahalf - 1;
        
        mreal beta = 2. * betahalf;
        
        mreal sum = 0.;

        mint n = bvh->primitive_count;
        
        mint data_dim = bvh->data_dim;
        mint nthreads = bvh->thread_count;
        
        // Dunno why "restrict" helps with P_data. It is actually a lie here.
        mreal const * const restrict A  = bvh->P_data[0];
        mreal const * const restrict X1 = bvh->P_data[1];
        mreal const * const restrict X2 = bvh->P_data[2];
        mreal const * const restrict X3 = bvh->P_data[3];
        mreal const * const restrict N1 = bvh->P_data[4];
        mreal const * const restrict N2 = bvh->P_data[5];
        mreal const * const restrict N3 = bvh->P_data[6];
        
        mreal const * const restrict B  = bvh->P_data[0];
        mreal const * const restrict Y1 = bvh->P_data[1];
        mreal const * const restrict Y2 = bvh->P_data[2];
        mreal const * const restrict Y3 = bvh->P_data[3];
        mreal const * const restrict M1 = bvh->P_data[4];
        mreal const * const restrict M2 = bvh->P_data[5];
        mreal const * const restrict M3 = bvh->P_data[6];
        
        #pragma omp parallel for num_threads( nthreads ) reduction( +: sum )
        for( mint i = 0; i < n ; ++i )
        {
            
            mint thread = omp_get_thread_num();
            
            mreal * const restrict U = &bvh->P_D_data[thread][0];
            mreal * const restrict V = &bvh->P_D_data[thread][0];
            
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
                
                V[ data_dim * j ] += a * (
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
                
                V[ data_dim * j + 1 ] -= a  * Z1;
                V[ data_dim * j + 2 ] -= a  * Z2;
                V[ data_dim * j + 3 ] -= a  * Z3;
                V[ data_dim * j + 4 ] += aG * v1;
                V[ data_dim * j + 5 ] += aG * v2;
                V[ data_dim * j + 6 ] += aG * v3;
            }// for( mint j = begin; j < T_n; ++j )
            
            sum += i_sum;
            
            U[ data_dim * i     ] +=  da;
            U[ data_dim * i + 1 ] += dx1;
            U[ data_dim * i + 2 ] += dx2;
            U[ data_dim * i + 3 ] += dx3;
            U[ data_dim * i + 4 ] += dn1;
            U[ data_dim * i + 5 ] += dn2;
            U[ data_dim * i + 6 ] += dn3;
            
        }// for( mint i = 0; i < S_n ; ++i )
        return sum;
    }; //DEnergy

} // namespace rsurfaces
