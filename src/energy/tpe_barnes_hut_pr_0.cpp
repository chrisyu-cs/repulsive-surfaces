#include "energy/tpe_barnes_hut_0_pr.h"
#include "bct_constructors.h"

namespace rsurfaces
{
    
    template<typename T1, typename T2>
    mreal TPEnergyBarnesHut0_Projectors::Energy(T1 alphahalf, T2 betahalf)
    {
        
        T2 minus_betahalf = -betahalf;
        mreal theta2 = theta*theta;
        
        mint nthreads = bvh->thread_count;
        
        // Dunno why "restrict" helps with P_data. It is actually a lie here when S = T.
        mreal const * const restrict P_A  = bvh->P_data[0];
        mreal const * const restrict P_X1 = bvh->P_data[1];
        mreal const * const restrict P_X2 = bvh->P_data[2];
        mreal const * const restrict P_X3 = bvh->P_data[3];
        mreal const * const restrict P_P11 = bvh->P_data[4];
        mreal const * const restrict P_P12 = bvh->P_data[5];
        mreal const * const restrict P_P13 = bvh->P_data[6];
        mreal const * const restrict P_P22 = bvh->P_data[7];
        mreal const * const restrict P_P23 = bvh->P_data[8];
        mreal const * const restrict P_P33 = bvh->P_data[9];

        mreal const * const restrict C_min1 = bvh->C_min[0];
        mreal const * const restrict C_min2 = bvh->C_min[1];
        mreal const * const restrict C_min3 = bvh->C_min[2];
        
        mreal const * const restrict C_max1 = bvh->C_max[0];
        mreal const * const restrict C_max2 = bvh->C_max[1];
        mreal const * const restrict C_max3 = bvh->C_max[2];
        
        mreal const * const restrict C_A  = bvh->C_data[0];
        mreal const * const restrict C_X1 = bvh->C_data[1];
        mreal const * const restrict C_X2 = bvh->C_data[2];
        mreal const * const restrict C_X3 = bvh->C_data[3];

        mint  const * const restrict C_left  = bvh->C_left;
        mint  const * const restrict C_right = bvh->C_right;
        mint  const * const restrict C_begin = bvh->C_begin;
        mint  const * const restrict C_end   = bvh->C_end;
        mreal const * const restrict C_r2    = bvh->C_squared_radius;
        
        mint  const * const restrict leaf = bvh->leaf_clusters;
        
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
                
                // Compute squared distance between bounding boxes.
                // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares
                
                mreal ymin1 = C_min1[C];
                mreal ymin2 = C_min2[C];
                mreal ymin3 = C_min3[C];
                
                mreal ymax1 = C_max1[C];
                mreal ymax2 = C_max2[C];
                mreal ymax3 = C_max3[C];
                
                mreal d1 = mymax( 0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1) );
                mreal d2 = mymax( 0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2) );
                mreal d3 = mymax( 0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3) );
                
                mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;

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
    mreal TPEnergyBarnesHut0_Projectors::DEnergy(T1 alphahalf, T2 betahalf)
    {
        
        T1 alphahalf_minus_1 = alphahalf - 1;
        T2 minus_betahalf_minus_1 = -betahalf - 1;
        
        mreal beta = 2. * betahalf;

        mreal theta2 = theta * theta;
        mreal sum = 0.;
        
        mint nthreads = bvh->thread_count;
        
        // Dunno why "restrict" helps with P_data. It is actually a lie here when S = T.
        mreal const * const restrict P_A  = bvh->P_data[0];
        mreal const * const restrict P_X1 = bvh->P_data[1];
        mreal const * const restrict P_X2 = bvh->P_data[2];
        mreal const * const restrict P_X3 = bvh->P_data[3];
        mreal const * const restrict P_P11 = bvh->P_data[4];
        mreal const * const restrict P_P12 = bvh->P_data[5];
        mreal const * const restrict P_P13 = bvh->P_data[6];
        mreal const * const restrict P_P22 = bvh->P_data[7];
        mreal const * const restrict P_P23 = bvh->P_data[8];
        mreal const * const restrict P_P33 = bvh->P_data[9];

        mreal const * const restrict C_min1 = bvh->C_min[0];
        mreal const * const restrict C_min2 = bvh->C_min[1];
        mreal const * const restrict C_min3 = bvh->C_min[2];
        
        mreal const * const restrict C_max1 = bvh->C_max[0];
        mreal const * const restrict C_max2 = bvh->C_max[1];
        mreal const * const restrict C_max3 = bvh->C_max[2];
        
        mreal const * const restrict C_A  = bvh->C_data[0];
        mreal const * const restrict C_X1 = bvh->C_data[1];
        mreal const * const restrict C_X2 = bvh->C_data[2];
        mreal const * const restrict C_X3 = bvh->C_data[3];

        
        mint  const * const restrict C_left  = bvh->C_left;
        mint  const * const restrict C_right = bvh->C_right;
        mint  const * const restrict C_begin = bvh->C_begin;
        mint  const * const restrict C_end   = bvh->C_end;
        mreal const * const restrict C_r2    = bvh->C_squared_radius;
        
        mint  const * const restrict leaf = bvh->leaf_clusters;
        
        A_Vector<A_Vector<mint>> thread_stack ( nthreads );
        
        #pragma omp parallel for num_threads( nthreads ) reduction( + : sum)
        for( mint k = 0; k < bvh->leaf_cluster_count; ++k )
        {
            mint thread = omp_get_thread_num();
            
            A_Vector<mint> * stack = &thread_stack[thread];
            
            mreal * const restrict P_U = &bvh->P_D_data[thread][0];
            mreal * const restrict C_U = &bvh->C_D_data[thread][0];
            
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
                
                // Compute squared distance between bounding boxes.
                // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares
                
                mreal ymin1 = C_min1[C];
                mreal ymin2 = C_min2[C];
                mreal ymin3 = C_min3[C];
                
                mreal ymax1 = C_max1[C];
                mreal ymax2 = C_max2[C];
                mreal ymax3 = C_max3[C];
                
                mreal d1 = mymax( 0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1) );
                mreal d2 = mymax( 0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2) );
                mreal d3 = mymax( 0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3) );
                
                mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;

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

    
    double TPEnergyBarnesHut0_Projectors::Value()
    {
        
        if( use_int )
        {
            mint int_alphahalf = std::round(alpha/2);
            mint int_betahalf = std::round(beta/2);
            return Energy( int_alphahalf, int_betahalf );
        }
        else
        {
            mreal real_alphahalf = alpha/2;
            mreal real_betahalf = beta/2;
            return Energy( real_alphahalf, real_betahalf );
        }
    } // Value

    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void TPEnergyBarnesHut0_Projectors::Update()
    {
        if (bvh)
        {
            delete bvh;
        }
        
        bvh = CreateOptimizedBVH_Projectors(mesh, geom);
    }

    // Get the mesh associated with this energy.
    MeshPtr TPEnergyBarnesHut0_Projectors::GetMesh()
    {
        return mesh;
    }

    // Get the geometry associated with this geometry.
    GeomPtr TPEnergyBarnesHut0_Projectors::GetGeom()
    {
        return geom;
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 TPEnergyBarnesHut0_Projectors::GetExponents()
    {
        return Vector2{alpha, beta};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *TPEnergyBarnesHut0_Projectors::GetBVH()
    {
        return bvh;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double TPEnergyBarnesHut0_Projectors::GetTheta()
    {
        return theta;
    }

    void TPEnergyBarnesHut0_Projectors::Differential( Eigen::MatrixXd &output )
    {
        
        auto V_coords = getVertexPositions( mesh, geom );
        auto primitives = getPrimitiveIndices( mesh, geom );
        
        mint vertex_count = V_coords.rows();
        mint dim = V_coords.cols();
        mint primitive_count = primitives.rows();
        mint primitive_length = primitives.cols();
        
        EigenMatrixRM P_D_data ( bvh->primitive_count , bvh->data_dim );
        
        if( bvh->data_dim != 10)
        {
            eprint("in TPEnergyBarnesHut0_Projectors::Differential: data_dim != 10");
        }
        
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
        
        AssembleDerivativeFromACPData( mesh, geom, P_D_data, output );
        
    } // Differential


//template<typename T1, typename T2>
//mreal TPEnergyBarnesHut0_Projectors::Energy(T1 alpha, T2 betahalf)
//{
//
//    T2 minus_betahalf = -betahalf;
//    mreal theta2 = theta*theta;
//
//    mint nthreads = bvh->thread_count;
//
//    // Dunno why "restrict" helps with P_data. It is actually a lie here when S = T.
//    mreal const * const restrict P_A  = bvh->P_data[0];
//    mreal const * const restrict P_X1 = bvh->P_data[1];
//    mreal const * const restrict P_X2 = bvh->P_data[2];
//    mreal const * const restrict P_X3 = bvh->P_data[3];
//    mreal const * const restrict P_N1 = bvh->P_data[4];
//    mreal const * const restrict P_N2 = bvh->P_data[5];
//    mreal const * const restrict P_N3 = bvh->P_data[6];
//
//    mreal const * const restrict C_min1 = bvh->C_min[0];
//    mreal const * const restrict C_min2 = bvh->C_min[1];
//    mreal const * const restrict C_min3 = bvh->C_min[2];
//
//    mreal const * const restrict C_max1 = bvh->C_max[0];
//    mreal const * const restrict C_max2 = bvh->C_max[1];
//    mreal const * const restrict C_max3 = bvh->C_max[2];
//
//    mreal const * const restrict C_A  = bvh->C_data[0];
//    mreal const * const restrict C_X1 = bvh->C_data[1];
//    mreal const * const restrict C_X2 = bvh->C_data[2];
//    mreal const * const restrict C_X3 = bvh->C_data[3];
//
//    mint  const * const restrict C_left  = bvh->C_left;
//    mint  const * const restrict C_right = bvh->C_right;
//    mint  const * const restrict C_begin = bvh->C_begin;
//    mint  const * const restrict C_end   = bvh->C_end;
//    mreal const * const restrict C_r2    = bvh->C_squared_radius;
//
//    mint  const * const restrict leaf = bvh->leaf_clusters;
//
//    A_Vector<A_Vector<mint>> thread_stack ( nthreads );
//
//    mreal sum = 0.;
//
//    #pragma omp parallel for num_threads( nthreads ) reduction( + : sum)
//    for( mint k = 0; k < bvh->leaf_cluster_count; ++k )
//    {
//        mint thread = omp_get_thread_num();
//
//        A_Vector<mint> * stack = &thread_stack[thread];
//
//        stack->clear();
//        stack->push_back(0);
//
//        mint l = leaf[k];
//        mint i_begin = C_begin[l];
//        mint i_end   = C_end[l];
//
//        mreal xmin1 = C_min1[l];
//        mreal xmin2 = C_min2[l];
//        mreal xmin3 = C_min3[l];
//
//        mreal xmax1 = C_max1[l];
//        mreal xmax2 = C_max2[l];
//        mreal xmax3 = C_max3[l];
//
//        mreal r2l = C_r2[l];
//
//        mreal local_sum = 0.;
//
//        while( !stack->empty() )
//        {
//            mint C = stack->back();
//            stack->pop_back();
//
//            mreal h2 = std::max(r2l, C_r2[C]);
//
//            // Compute squared distance between bounding boxes.
//            // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares
//
//            mreal ymin1 = C_min1[C];
//            mreal ymin2 = C_min2[C];
//            mreal ymin3 = C_min3[C];
//
//            mreal ymax1 = C_max1[C];
//            mreal ymax2 = C_max2[C];
//            mreal ymax3 = C_max3[C];
//
//            mreal d1 = mymax( 0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1) );
//            mreal d2 = mymax( 0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2) );
//            mreal d3 = mymax( 0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3) );
//
//            mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;
//
//            if( h2 < theta2 * R2 )
//            {
//                mreal b  = C_A [C];
//                mreal y1 = C_X1[C];
//                mreal y2 = C_X2[C];
//                mreal y3 = C_X3[C];
//
//                mreal local_local_sum = 0.;
//
//                for( mint i = i_begin; i < i_end; ++i )
//                {
//                    mreal a  = P_A [i];
//                    mreal x1 = P_X1[i];   mreal n1 = P_N1[i];
//                    mreal x2 = P_X2[i];   mreal n2 = P_N2[i];
//                    mreal x3 = P_X3[i];   mreal n3 = P_N3[i];
//
//                    mreal v1 = y1 - x1;
//                    mreal v2 = y2 - x2;
//                    mreal v3 = y3 - x3;
//
//                    mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
//                    mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 ;
//                    local_local_sum += a * mypow( fabs(rCosPhi), alpha ) * mypow( r2, minus_betahalf );
//                }
//                local_sum += local_local_sum  * b;
//
//            }
//            else
//            {
//                mint left  = C_left[C];
//                mint right = C_right[C];
//                if( left >= 0 && right >= 0 )
//                {
//                    stack->push_back( right );
//                    stack->push_back( left  );
//                }
//                else
//                {
//                    // near field loop
//                    mint j_begin = C_begin[C];
//                    mint j_end   = C_end[C];
//
//                    for( mint i = i_begin; i < i_end; ++i )
//                    {
//                        mreal a  = P_A [i];
//                        mreal x1 = P_X1[i];   mreal n1 = P_N1[i];
//                        mreal x2 = P_X2[i];   mreal n2 = P_N2[i];
//                        mreal x3 = P_X3[i];   mreal n3 = P_N3[i];
//
//                        mreal local_local_sum = 0.;
//
////                        #pragma omp simd aligned( P_A, P_X1, P_X3 : ALIGN )
//                        for( mint j = j_begin; j < j_end; ++j )
//                        {
//                            if ( i != j )
//                            {
//                                mreal b  = P_A [j];
//                                mreal v1 = P_X1[j] - x1;
//                                mreal v2 = P_X2[j] - x2;
//                                mreal v3 = P_X3[j] - x3;
//
//                                mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
//                                mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 ;
//
//                                local_local_sum += mypow( fabs(rCosPhi), alpha ) * mypow( r2, minus_betahalf ) * b;
//                            }
//                        }
//
//                        local_sum += a * local_local_sum;
//                    }
//                }
//            }
//        }
//
//        sum += local_sum;
//    }
//    return sum;
//}; // Energy
//
//
//template<typename T1, typename T2>
//mreal TPEnergyBarnesHut0_Projectors::DEnergy(T1 alpha, T2 betahalf)
//{
//
//    T1 alpha_minus_2 = alpha - 2;
//    T2 minus_betahalf_minus_1 = -betahalf - 1;
//
//    mreal beta = 2. * betahalf;
//    mreal theta2 = theta * theta;
//    mreal sum = 0.;
//
//    mint nthreads = bvh->thread_count;
//
//    // Dunno why "restrict" helps with P_data. It is actually a lie here when S = T.
//    mreal const * const restrict P_A  = bvh->P_data[0];
//    mreal const * const restrict P_X1 = bvh->P_data[1];
//    mreal const * const restrict P_X2 = bvh->P_data[2];
//    mreal const * const restrict P_X3 = bvh->P_data[3];
//    mreal const * const restrict P_N1 = bvh->P_data[4];
//    mreal const * const restrict P_N2 = bvh->P_data[5];
//    mreal const * const restrict P_N3 = bvh->P_data[6];
//
//    mreal const * const restrict C_min1 = bvh->C_min[0];
//    mreal const * const restrict C_min2 = bvh->C_min[1];
//    mreal const * const restrict C_min3 = bvh->C_min[2];
//
//    mreal const * const restrict C_max1 = bvh->C_max[0];
//    mreal const * const restrict C_max2 = bvh->C_max[1];
//    mreal const * const restrict C_max3 = bvh->C_max[2];
//
//    mreal const * const restrict C_A  = bvh->C_data[0];
//    mreal const * const restrict C_X1 = bvh->C_data[1];
//    mreal const * const restrict C_X2 = bvh->C_data[2];
//    mreal const * const restrict C_X3 = bvh->C_data[3];
//
//
//    mint  const * const restrict C_left  = bvh->C_left;
//    mint  const * const restrict C_right = bvh->C_right;
//    mint  const * const restrict C_begin = bvh->C_begin;
//    mint  const * const restrict C_end   = bvh->C_end;
//    mreal const * const restrict C_r2    = bvh->C_squared_radius;
//
//    mint  const * const restrict leaf = bvh->leaf_clusters;
//
//    A_Vector<A_Vector<mint>> thread_stack ( nthreads );
//
//    #pragma omp parallel for num_threads( nthreads ) reduction( + : sum)
//    for( mint k = 0; k < bvh->leaf_cluster_count; ++k )
//    {
//        mint thread = omp_get_thread_num();
//
//        A_Vector<mint> * stack = &thread_stack[thread];
//
//        mreal * const restrict P_U = &bvh->P_D_data[thread][0];
//        mreal * const restrict C_U = &bvh->C_D_data[thread][0];
//
//        stack->clear();
//        stack->push_back(0);
//
//        mint l = leaf[k];
//        mint i_begin = C_begin[l];
//        mint i_end   = C_end[l];
//
//        mreal xmin1 = C_min1[l];
//        mreal xmin2 = C_min2[l];
//        mreal xmin3 = C_min3[l];
//
//        mreal xmax1 = C_max1[l];
//        mreal xmax2 = C_max2[l];
//        mreal xmax3 = C_max3[l];
//
//        mreal r2l = C_r2[l];
//
//        while( !stack->empty() )
//        {
//            mint C = stack->back();
//            stack->pop_back();
//
//            mreal h2 = std::max(r2l, C_r2[C]);
//
//            // Compute squared distance between bounding boxes.
//            // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares
//
//            mreal ymin1 = C_min1[C];
//            mreal ymin2 = C_min2[C];
//            mreal ymin3 = C_min3[C];
//
//            mreal ymax1 = C_max1[C];
//            mreal ymax2 = C_max2[C];
//            mreal ymax3 = C_max3[C];
//
//            mreal d1 = mymax( 0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1) );
//            mreal d2 = mymax( 0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2) );
//            mreal d3 = mymax( 0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3) );
//
//            mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;
//
//            if( h2 < theta2 * R2 )
//            {
//                mreal b  = C_A [C];
//                mreal y1 = C_X1[C];
//                mreal y2 = C_X2[C];
//                mreal y3 = C_X3[C];
//
//                for( mint i = i_begin; i < i_end; ++i )
//                {
//                    mreal a  = P_A [i];
//                    mreal x1 = P_X1[i];   mreal n1 = P_N1[i];
//                    mreal x2 = P_X2[i];   mreal n2 = P_N2[i];
//                    mreal x3 = P_X3[i];   mreal n3 = P_N3[i];
//
//                    mreal v1 = y1 - x1;
//                    mreal v2 = y2 - x2;
//                    mreal v3 = y3 - x3;
//
//                    mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
//                    mreal r2      = v1 * v1 + v2 * v2 + v3 * v3;
//
//                    mreal rBetaMinus2 = mypow( r2, minus_betahalf_minus_1 );
//                    mreal rBeta = rBetaMinus2 * r2;
//
//                    mreal rCosPhiAlphaMinus1 = mypow( fabs(rCosPhi), alpha_minus_2 ) * rCosPhi;
//                    mreal rCosPhiAlpha = rCosPhiAlphaMinus1 * rCosPhi;
//
//                    mreal Num = rCosPhiAlpha;
//                    mreal factor0 = rBeta * alpha;
//                    mreal density = rBeta * Num;
//                    sum += a * b * density;
//
//                    mreal F = factor0 * rCosPhiAlphaMinus1;
//                    mreal H = beta * rBetaMinus2 * Num;
//
//                    mreal bF = b * F;
//
//                    mreal Z1 = ( - n1 * F + v1 * H );
//                    mreal Z2 = ( - n2 * F + v2 * H );
//                    mreal Z3 = ( - n3 * F + v3 * H );
//
//                    P_U[ 7 * i + 0 ] += b * (
//                                             density
//                                             +
//                                             F * ( n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3) )
//                                             -
//                                             H * ( v1 * x1 + v2 * x2 + v3 * x3 )
//                                             );
//                    C_U[ 7 * C + 0 ] += a * (
//                                             density
//                                             -
//                                             F * ( n1 * y1 + n2 * y2 + n3 * y3 )
//                                             +
//                                             H * ( v1 * y1 + v2 * y2 + v3 * y3 )
//                                             );
//
//
//                    P_U[ 7 * i + 1 ] += b  * Z1;
//                    P_U[ 7 * i + 2 ] += b  * Z2;
//                    P_U[ 7 * i + 3 ] += b  * Z3;
//                    P_U[ 7 * i + 4 ] += bF * v1;
//                    P_U[ 7 * i + 5 ] += bF * v2;
//                    P_U[ 7 * i + 6 ] += bF * v3;
//
//                    C_U[ 7 * C + 1 ] -= a  * Z1;
//                    C_U[ 7 * C + 2 ] -= a  * Z2;
//                    C_U[ 7 * C + 3 ] -= a  * Z3;
//                }
//            }
//            else
//            {
//                mint left  = C_left[C];
//                mint right = C_right[C];
//                if( left >= 0 && right >= 0 )
//                {
//                    stack->push_back( right );
//                    stack->push_back( left  );
//                }
//                else
//                {
//                    // near field loop
//                    mint j_begin = C_begin[C];
//                    mint j_end   = C_end[C];
//
//                    for( mint i = i_begin; i < i_end; ++i )
//                    {
//                        mreal a  = P_A [i];
//                        mreal x1 = P_X1[i];   mreal n1 = P_N1[i];
//                        mreal x2 = P_X2[i];   mreal n2 = P_N2[i];
//                        mreal x3 = P_X3[i];   mreal n3 = P_N3[i];
//
//                        for( mint j = j_begin; j < j_end; ++j )
//                        {
//                            if ( i != j )
//                            {
//                                mreal b  = P_A [j];
//                                mreal y1 = P_X1[j];
//                                mreal y2 = P_X2[j];
//                                mreal y3 = P_X3[j];
//
//                                mreal v1 = y1 - x1;
//                                mreal v2 = y2 - x2;
//                                mreal v3 = y3 - x3;
//
//                                mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
//                                mreal r2      = v1 * v1 + v2 * v2 + v3 * v3;
//
//                                mreal rBetaMinus2 = mypow( r2, minus_betahalf_minus_1 );
//                                mreal rBeta = rBetaMinus2 * r2;
//
//                                mreal rCosPhiAlphaMinus1 = mypow( fabs(rCosPhi), alpha_minus_2 ) * rCosPhi;
//                                mreal rCosPhiAlpha = rCosPhiAlphaMinus1 * rCosPhi;
//
//                                mreal Num = rCosPhiAlpha;
//                                mreal factor0 = rBeta * alpha;
//                                mreal density = rBeta * Num;
//                                sum += a * b * density;
//
//                                mreal F = factor0 * rCosPhiAlphaMinus1;
//                                mreal H = beta * rBetaMinus2 * Num;
//
//                                mreal bF = b * F;
//
//                                mreal Z1 = ( - n1 * F + v1 * H );
//                                mreal Z2 = ( - n2 * F + v2 * H );
//                                mreal Z3 = ( - n3 * F + v3 * H );
//
//                                P_U[ 7 * i + 0 ] += b * (
//                                                         density
//                                                         +
//                                                         F * ( n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3) )
//                                                         -
//                                                         H * ( v1 * x1 + v2 * x2 + v3 * x3 )
//                                                         );
//                                P_U[ 7 * j + 0 ] += a * (
//                                                         density
//                                                         -
//                                                         F * ( n1 * y1 + n2 * y2 + n3 * y3 )
//                                                         +
//                                                         H * ( v1 * y1 + v2 * y2 + v3 * y3 )
//                                                         );
//
//
//                                P_U[ 7 * i + 1 ] += b  * Z1;
//                                P_U[ 7 * i + 2 ] += b  * Z2;
//                                P_U[ 7 * i + 3 ] += b  * Z3;
//                                P_U[ 7 * i + 4 ] += bF * v1;
//                                P_U[ 7 * i + 5 ] += bF * v2;
//                                P_U[ 7 * i + 6 ] += bF * v3;
//
//                                P_U[ 7 * j + 1 ] -= a  * Z1;
//                                P_U[ 7 * j + 2 ] -= a  * Z2;
//                                P_U[ 7 * j + 3 ] -= a  * Z3;
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//    return sum;
//}; // DEnergy


} // namespace rsurfaces
