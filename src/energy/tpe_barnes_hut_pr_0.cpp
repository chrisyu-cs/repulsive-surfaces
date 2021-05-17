#include "energy/tpe_barnes_hut_pr_0.h"
#include "bct_constructors.h"

namespace rsurfaces
{
    
    template<typename T1, typename T2>
    mreal TPEnergyBarnesHut_Projectors0::Energy(T1 alphahalf, T2 betahalf)
    {
        
        T2 minus_betahalf = -betahalf;
        mreal theta2 = theta*theta;
        
        mint nthreads = bvh->thread_count;
        
        // Dunno why "restrict" helps with P_near. It is actually a lie here when S = T.
        mreal const * restrict const P_A  = bvh->P_near[0];
        mreal const * restrict const P_X1 = bvh->P_near[1];
        mreal const * restrict const P_X2 = bvh->P_near[2];
        mreal const * restrict const P_X3 = bvh->P_near[3];
        mreal const * restrict const P_P11 = bvh->P_near[4];
        mreal const * restrict const P_P12 = bvh->P_near[5];
        mreal const * restrict const P_P13 = bvh->P_near[6];
        mreal const * restrict const P_P22 = bvh->P_near[7];
        mreal const * restrict const P_P23 = bvh->P_near[8];
        mreal const * restrict const P_P33 = bvh->P_near[9];

        mreal const * restrict const C_min1 = bvh->C_min[0];
        mreal const * restrict const C_min2 = bvh->C_min[1];
        mreal const * restrict const C_min3 = bvh->C_min[2];
        
        mreal const * restrict const C_max1 = bvh->C_max[0];
        mreal const * restrict const C_max2 = bvh->C_max[1];
        mreal const * restrict const C_max3 = bvh->C_max[2];
        
        mreal const * restrict const C_A  = bvh->C_far[0];
        mreal const * restrict const C_X1 = bvh->C_far[1];
        mreal const * restrict const C_X2 = bvh->C_far[2];
        mreal const * restrict const C_X3 = bvh->C_far[3];

        mint  const * restrict const C_left  = bvh->C_left;
        mint  const * restrict const C_right = bvh->C_right;
        mint  const * restrict const C_begin = bvh->C_begin;
        mint  const * restrict const C_end   = bvh->C_end;
        mreal const * restrict const C_r2    = bvh->C_squared_radius;
        
        mint  const * restrict const leaf = bvh->leaf_clusters;
        
        A_Vector<A_Vector<mint>> thread_stack ( nthreads );
        
        mreal sum = 0.;
        
        #pragma omp parallel for num_threads( nthreads ) reduction( + : sum)
        for( mint k = 0; k < bvh->leaf_cluster_count; ++k )
        {
            mint thread = omp_get_thread_num();
            
            A_Vector<mint> * stack = &thread_stack[thread];
            
            stack->clear();
            stack->push_back(0);
            
            mint l = leaf[k];
            mint i_begin = C_begin[l];
            mint i_end   = C_end[l];
            
            mreal xmin1 = C_min1[l];
            mreal xmin2 = C_min2[l];
            mreal xmin3 = C_min3[l];
            
            mreal xmax1 = C_max1[l];
            mreal xmax2 = C_max2[l];
            mreal xmax3 = C_max3[l];
            
            mreal r2l = C_r2[l];
            
            mreal local_sum = 0.;
            
            while( !stack->empty() )
            {
                mint C = stack->back();
                stack->pop_back();
                
                mreal h2 = std::max(r2l, C_r2[C]);
                
                mreal R2 = SquaredBoxMinDistance( xmin1,     xmin2,     xmin3,     xmax1,     xmax2,     xmax3,
                                                  C_min1[C], C_min2[C], C_min3[C], C_max1[C], C_max2[C], C_max3[C]);

                if( h2 < theta2 * R2 )
                {
                    mreal b  = C_A [C];
                    mreal y1 = C_X1[C];
                    mreal y2 = C_X2[C];
                    mreal y3 = C_X3[C];
                    
                    mreal local_local_sum = 0.;
                    
                    for( mint i = i_begin; i < i_end; ++i )
                    {
                        mreal a  = P_A [i];
                        mreal x1 = P_X1[i];
                        mreal x2 = P_X2[i];
                        mreal x3 = P_X3[i];
                        mreal p11 = P_P11[i];
                        mreal p12 = P_P12[i];
                        mreal p13 = P_P13[i];
                        mreal p22 = P_P22[i];
                        mreal p23 = P_P23[i];
                        mreal p33 = P_P33[i];
                        
                        mreal v1 = y1 - x1;
                        mreal v2 = y2 - x2;
                        mreal v3 = y3 - x3;
                        
                        mreal rCosPhi2 = v1*(p11*v1 + p12*v2 + p13*v3) + v2*(p12*v1 + p22*v2 + p23*v3) + v3*(p13*v1 + p23*v2 + p33*v3);
                        mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 ;
                        local_local_sum += a * mypow( fabs(rCosPhi2), alphahalf ) * mypow( r2, minus_betahalf );
                    }
                    local_sum += local_local_sum  * b;
                    
                }
                else
                {
                    mint left  = C_left[C];
                    mint right = C_right[C];
                    if( left >= 0 && right >= 0 )
                    {
                        stack->push_back( right );
                        stack->push_back( left  );
                    }
                    else
                    {
                        // near field loop
                        mint j_begin = C_begin[C];
                        mint j_end   = C_end[C];

                        for( mint i = i_begin; i < i_end; ++i )
                        {
                            mreal a  = P_A [i];
                            mreal x1 = P_X1[i];
                            mreal x2 = P_X2[i];
                            mreal x3 = P_X3[i];
                            mreal p11 = P_P11[i];
                            mreal p12 = P_P12[i];
                            mreal p13 = P_P13[i];
                            mreal p22 = P_P22[i];
                            mreal p23 = P_P23[i];
                            mreal p33 = P_P33[i];
                            
                            mreal local_local_sum = 0.;
                            
    //                        #pragma omp simd aligned( P_A, P_X1, P_X3 : ALIGN )
                            for( mint j = j_begin; j < j_end; ++j )
                            {
                                if ( i != j )
                                {
                                    mreal b  = P_A [j];
                                    mreal v1 = P_X1[j] - x1;
                                    mreal v2 = P_X2[j] - x2;
                                    mreal v3 = P_X3[j] - x3;
                                    
                                    
                                    mreal rCosPhi2 = v1*(p11*v1 + p12*v2 + p13*v3) + v2*(p12*v1 + p22*v2 + p23*v3) + v3*(p13*v1 + p23*v2 + p33*v3);
                                    mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 ;
                                    
                                    local_local_sum += mypow( fabs(rCosPhi2), alphahalf ) * mypow( r2, minus_betahalf ) * b;
                                }
                            }
                            
                            local_sum += a * local_local_sum;
                        }
                    }
                }
            }
            
            sum += local_sum;
        }
        return sum;
    }; // Energy


    template<typename T1, typename T2>
    mreal TPEnergyBarnesHut_Projectors0::DEnergy(T1 alphahalf, T2 betahalf)
    {
        
        T1 alphahalf_minus_1 = alphahalf - 1;
        T2 minus_betahalf_minus_1 = -betahalf - 1;
        
        mreal beta = 2. * betahalf;

        mreal theta2 = theta * theta;
        mreal sum = 0.;
        
        mint nthreads = bvh->thread_count;
        
        // Dunno why "restrict" helps with P_near. It is actually a lie here when S = T.
        mreal const * restrict const P_A  = bvh->P_near[0];
        mreal const * restrict const P_X1 = bvh->P_near[1];
        mreal const * restrict const P_X2 = bvh->P_near[2];
        mreal const * restrict const P_X3 = bvh->P_near[3];
        mreal const * restrict const P_P11 = bvh->P_near[4];
        mreal const * restrict const P_P12 = bvh->P_near[5];
        mreal const * restrict const P_P13 = bvh->P_near[6];
        mreal const * restrict const P_P22 = bvh->P_near[7];
        mreal const * restrict const P_P23 = bvh->P_near[8];
        mreal const * restrict const P_P33 = bvh->P_near[9];

        mreal const * restrict const C_min1 = bvh->C_min[0];
        mreal const * restrict const C_min2 = bvh->C_min[1];
        mreal const * restrict const C_min3 = bvh->C_min[2];
        
        mreal const * restrict const C_max1 = bvh->C_max[0];
        mreal const * restrict const C_max2 = bvh->C_max[1];
        mreal const * restrict const C_max3 = bvh->C_max[2];
        
        mreal const * restrict const C_A  = bvh->C_far[0];
        mreal const * restrict const C_X1 = bvh->C_far[1];
        mreal const * restrict const C_X2 = bvh->C_far[2];
        mreal const * restrict const C_X3 = bvh->C_far[3];

        
        mint  const * restrict const C_left  = bvh->C_left;
        mint  const * restrict const C_right = bvh->C_right;
        mint  const * restrict const C_begin = bvh->C_begin;
        mint  const * restrict const C_end   = bvh->C_end;
        mreal const * restrict const C_r2    = bvh->C_squared_radius;
        
        mint  const * restrict const leaf = bvh->leaf_clusters;
        
        A_Vector<A_Vector<mint>> thread_stack ( nthreads );
        
        #pragma omp parallel for num_threads( nthreads ) reduction( + : sum)
        for( mint k = 0; k < bvh->leaf_cluster_count; ++k )
        {
            mint thread = omp_get_thread_num();
            
            A_Vector<mint> * stack = &thread_stack[thread];
            
            mreal * restrict const P_U = &bvh->P_D_near[thread][0];
            mreal * restrict const C_U = &bvh->C_D_far[thread][0];
            
            stack->clear();
            stack->push_back(0);
            
            mint l = leaf[k];
            mint i_begin = C_begin[l];
            mint i_end   = C_end[l];
            
            mreal xmin1 = C_min1[l];
            mreal xmin2 = C_min2[l];
            mreal xmin3 = C_min3[l];
            
            mreal xmax1 = C_max1[l];
            mreal xmax2 = C_max2[l];
            mreal xmax3 = C_max3[l];
            
            mreal r2l = C_r2[l];
            
            while( !stack->empty() )
            {
                mint C = stack->back();
                stack->pop_back();
                
                mreal h2 = std::max(r2l, C_r2[C]);
                
                mreal R2 = SquaredBoxMinDistance( xmin1,     xmin2,     xmin3,     xmax1,     xmax2,     xmax3,
                                                  C_min1[C], C_min2[C], C_min3[C], C_max1[C], C_max2[C], C_max3[C]);

                if( h2 < theta2 * R2 )
                {
                    mreal b  = C_A [C];
                    mreal y1 = C_X1[C];
                    mreal y2 = C_X2[C];
                    mreal y3 = C_X3[C];
                    
                    mreal local_sum = 0.;
                    
                    for( mint i = i_begin; i < i_end; ++i )
                    {
                        mreal a  = P_A [i];
                        mreal x1 = P_X1[i];
                        mreal x2 = P_X2[i];
                        mreal x3 = P_X3[i];
                        mreal p11 = P_P11[i];
                        mreal p12 = P_P12[i];
                        mreal p13 = P_P13[i];
                        mreal p22 = P_P22[i];
                        mreal p23 = P_P23[i];
                        mreal p33 = P_P33[i];
                        
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
                        
                        
                        mreal rCosPhiAlphaMinus2 = mypow( fabs(rCosPhi2), alphahalf_minus_1);
                        mreal rMinusBetaMinus2 = mypow( r2, minus_betahalf_minus_1 );
                        
                        mreal rMinusBeta = rMinusBetaMinus2 * r2;
                        mreal rCosPhiAlpha = rCosPhiAlphaMinus2 * rCosPhi2;
                        mreal Num = rCosPhiAlpha;
                    
                        mreal E = Num * rMinusBeta;
                        local_sum += a * b * E;
                        
                        mreal factor = alphahalf * rMinusBeta;
                        mreal F = factor * rCosPhiAlphaMinus2;
                        mreal H = - beta * rMinusBetaMinus2 * Num;
                        
                        mreal bF = b * F;
                        
                        mreal dEdv1 = 2. * (F * Pv1 ) + H * v1;
                        mreal dEdv2 = 2. * (F * Pv2 ) + H * v2;
                        mreal dEdv3 = 2. * (F * Pv3 ) + H * v3;
                        
                        P_U[ 10 * i + 0 ] += b * ( E + dEdv1 * x1 + dEdv2 * x2 + dEdv3 * x3 - factor * rCosPhiAlpha );
                        P_U[ 10 * i + 1 ] -= b * dEdv1;
                        P_U[ 10 * i + 2 ] -= b * dEdv2;
                        P_U[ 10 * i + 3 ] -= b * dEdv3;
                        P_U[ 10 * i + 4 ] += bF * v11;
                        P_U[ 10 * i + 5 ] += bF * v12;
                        P_U[ 10 * i + 6 ] += bF * v13;
                        P_U[ 10 * i + 7 ] += bF * v22;
                        P_U[ 10 * i + 8 ] += bF * v23;
                        P_U[ 10 * i + 9 ] += bF * v33;
                        
                        C_U[ 10 * C + 0 ] += a * ( E - dEdv1 * y1 - dEdv2 * y2 - dEdv3 * y3 );
                        C_U[ 10 * C + 1 ] += a * dEdv1;
                        C_U[ 10 * C + 2 ] += a * dEdv2;
                        C_U[ 10 * C + 3 ] += a * dEdv3;
                    }
                    sum += local_sum;
                }
                else
                {
                    mint left  = C_left[C];
                    mint right = C_right[C];
                    if( left >= 0 && right >= 0 )
                    {
                        stack->push_back( right );
                        stack->push_back( left  );
                    }
                    else
                    {
                        // near field loop
                        mint j_begin = C_begin[C];
                        mint j_end   = C_end[C];
                        
                        mreal local_sum = 0.;
                        for( mint i = i_begin; i < i_end; ++i )
                        {
                            mreal a  = P_A [i];
                            mreal x1 = P_X1[i];
                            mreal x2 = P_X2[i];
                            mreal x3 = P_X3[i];
                            mreal p11 = P_P11[i];
                            mreal p12 = P_P12[i];
                            mreal p13 = P_P13[i];
                            mreal p22 = P_P22[i];
                            mreal p23 = P_P23[i];
                            mreal p33 = P_P33[i];
                            
                            for( mint j = j_begin; j < j_end; ++j )
                            {
                                if ( i != j )
                                {
                                    mreal b  = P_A [j];
                                    mreal y1 = P_X1[j];
                                    mreal y2 = P_X2[j];
                                    mreal y3 = P_X3[j];
                                    
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
                                                                        
                                    mreal rCosPhiAlphaMinus2 = mypow( fabs(rCosPhi2), alphahalf_minus_1);
                                    mreal rMinusBetaMinus2 = mypow( r2, minus_betahalf_minus_1 );
                                    
                                    mreal rMinusBeta = rMinusBetaMinus2 * r2;
                                    mreal rCosPhiAlpha = rCosPhiAlphaMinus2 * rCosPhi2;
                                    mreal Num = rCosPhiAlpha;
                                
                                    mreal E = Num * rMinusBeta;
                                    local_sum += a * b * E;
                                    
                                    mreal factor = alphahalf * rMinusBeta;
                                    mreal F = factor * rCosPhiAlphaMinus2;
                                    mreal H = - beta * rMinusBetaMinus2 * Num;
                                    
                                    mreal bF = b * F;
                                    
                                    mreal dEdv1 = 2. * (F * Pv1 ) + H * v1;
                                    mreal dEdv2 = 2. * (F * Pv2 ) + H * v2;
                                    mreal dEdv3 = 2. * (F * Pv3 ) + H * v3;
                                    
                                    P_U[ 10 * i + 0 ] += b * ( E + dEdv1 * x1 + dEdv2 * x2 + dEdv3 * x3 - factor * rCosPhiAlpha );
                                    P_U[ 10 * i + 1 ] -= b * dEdv1;
                                    P_U[ 10 * i + 2 ] -= b * dEdv2;
                                    P_U[ 10 * i + 3 ] -= b * dEdv3;
                                    P_U[ 10 * i + 4 ] += bF * v11;
                                    P_U[ 10 * i + 5 ] += bF * v12;
                                    P_U[ 10 * i + 6 ] += bF * v13;
                                    P_U[ 10 * i + 7 ] += bF * v22;
                                    P_U[ 10 * i + 8 ] += bF * v23;
                                    P_U[ 10 * i + 9 ] += bF * v33;
                                    
                                    P_U[ 10 * j + 0 ] += a * ( E - dEdv1 * y1 - dEdv2 * y2 - dEdv3 * y3);
                                    P_U[ 10 * j + 1 ] += a * dEdv1;
                                    P_U[ 10 * j + 2 ] += a * dEdv2;
                                    P_U[ 10 * j + 3 ] += a * dEdv3;
                                }
                            }
                        }
                        sum += local_sum;
                    }
                }
            }
        }
        return sum;
    }; // DEnergy

    
    double TPEnergyBarnesHut_Projectors0::Value()
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

    void TPEnergyBarnesHut_Projectors0::Differential( Eigen::MatrixXd &output )
    {
        if( bvh->near_dim != 10)
        {
            eprint("in TPEnergyBarnesHut_Projectors0::Differential: near_dim != 10");
        }
        if( bvh->far_dim != 10)
        {
            eprint("in TPEnergyBarnesHut_Projectors0::Differential: far_dim != 10");
        }
        
        EigenMatrixRM P_D_near( bvh->primitive_count , bvh->near_dim );
        EigenMatrixRM P_D_far ( bvh->primitive_count , bvh->far_dim );
        
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
        
        bvh->CollectDerivatives( P_D_near.data(), P_D_far.data() );
                
        AssembleDerivativeFromACPData( mesh, geom, P_D_near, output, weight );
        AssembleDerivativeFromACPData( mesh, geom, P_D_far, output, weight );
        
    } // Differential
    
    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void TPEnergyBarnesHut_Projectors0::Update()
    {
        if (bvh)
        {
            delete bvh;
        }
        
        bvh = CreateOptimizedBVH_Projectors(mesh, geom);
    }

    // Get the mesh associated with this energy.
    MeshPtr TPEnergyBarnesHut_Projectors0::GetMesh()
    {
        return mesh;
    }

    // Get the geometry associated with this geometry.
    GeomPtr TPEnergyBarnesHut_Projectors0::GetGeom()
    {
        return geom;
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 TPEnergyBarnesHut_Projectors0::GetExponents()
    {
        return Vector2{alpha, beta};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *TPEnergyBarnesHut_Projectors0::GetBVH()
    {
        return bvh;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double TPEnergyBarnesHut_Projectors0::GetTheta()
    {
        return theta;
    }

} // namespace rsurfaces
