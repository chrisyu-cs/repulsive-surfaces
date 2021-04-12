#include "energy/tpe_barnes_hut_0.h"
#include "bct_constructors.h"

namespace rsurfaces
{
    template<typename T1, typename T2>
    mreal TPEnergyBarnesHut0::Energy(T1 alpha, T2 betahalf)
    {
        ptic("TPEnergyBarnesHut0::Energy");
        
        T2 minus_betahalf = -betahalf;
        mreal theta2 = theta*theta;
        mreal sum = 0.;
        
        mint nthreads = bvh->thread_count;
        
        // Dunno why "restrict" helps with P_near. It is actually a lie here when S = T.
        mreal const * restrict const P_A  = bvh->P_near[0];
        mreal const * restrict const P_X1 = bvh->P_near[1];
        mreal const * restrict const P_X2 = bvh->P_near[2];
        mreal const * restrict const P_X3 = bvh->P_near[3];
        mreal const * restrict const P_N1 = bvh->P_near[4];
        mreal const * restrict const P_N2 = bvh->P_near[5];
        mreal const * restrict const P_N3 = bvh->P_near[6];

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
        
        
        #pragma omp parallel for num_threads( nthreads ) reduction( + : sum ) RAGGED_SCHEDULE
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
                    
                    #pragma omp simd aligned (P_A, P_X1, P_X2, P_X3, P_N1, P_N2, P_N3: ALIGN ) reduction( + : local_local_sum)
                    for( mint i = i_begin; i < i_end; ++i )
                    {
                        mreal a  = P_A [i];
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
                        mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 ;
                        local_local_sum += a * mypow( fabs(rCosPhi), alpha ) * mypow( r2, minus_betahalf );
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

//                        #pragma omp simd aligned( P_A, P_X1, P_X2, P_X3 : ALIGN ) collapse(2)
                        for( mint i = i_begin; i < i_end; ++i )
                        {
                            mreal a  = P_A [i];
                            mreal x1 = P_X1[i];
                            mreal x2 = P_X2[i];
                            mreal x3 = P_X3[i];
                            mreal n1 = P_N1[i];
                            mreal n2 = P_N2[i];
                            mreal n3 = P_N3[i];
                            
                            mreal local_local_sum = 0.;
                            
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
                                    
                                    mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                                    mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 ;
                                    
                                    local_local_sum += mypow( fabs(rCosPhi), alpha ) * mypow( r2, minus_betahalf ) * b;
                                }
                            }
                            
                            local_sum += a * local_local_sum;
                        }
                    }
                }
            }
            
            sum += local_sum;
        }
        
        ptoc("TPEnergyBarnesHut0::Energy");
        return sum;
    }; // Energy


    template<typename T1, typename T2>
    mreal TPEnergyBarnesHut0::DEnergy(T1 alpha, T2 betahalf)
    {
        ptic("TPEnergyBarnesHut0::DEnergy");
        T1 alpha_minus_2 = alpha - 2;
        T2 minus_betahalf_minus_1 = -betahalf - 1;
        
        mreal beta = 2. * betahalf;
        mreal theta2 = theta * theta;
        mreal sum = 0.;
        
        mint far_dim = bvh->far_dim;
        
        mint nthreads = bvh->thread_count;
        
        // Dunno why "restrict" helps with P_near. It is actually a lie here when S = T.
        mreal const * restrict const P_A  = bvh->P_near[0];
        mreal const * restrict const P_X1 = bvh->P_near[1];
        mreal const * restrict const P_X2 = bvh->P_near[2];
        mreal const * restrict const P_X3 = bvh->P_near[3];
        mreal const * restrict const P_N1 = bvh->P_near[4];
        mreal const * restrict const P_N2 = bvh->P_near[5];
        mreal const * restrict const P_N3 = bvh->P_near[6];

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
        
        #pragma omp parallel for num_threads( nthreads ) reduction( + : sum ) RAGGED_SCHEDULE
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
                    
                    for( mint i = i_begin; i < i_end; ++i )
                    {
                        mreal a  = P_A [i];
                        mreal x1 = P_X1[i];   mreal n1 = P_N1[i];
                        mreal x2 = P_X2[i];   mreal n2 = P_N2[i];
                        mreal x3 = P_X3[i];   mreal n3 = P_N3[i];
                        
                        mreal v1 = y1 - x1;
                        mreal v2 = y2 - x2;
                        mreal v3 = y3 - x3;
                        
                        mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                        mreal r2      = v1 * v1 + v2 * v2 + v3 * v3;
                        
                        mreal rBetaMinus2 = mypow( r2, minus_betahalf_minus_1 );
                        mreal rBeta = rBetaMinus2 * r2;
                        
                        mreal rCosPhiAlphaMinus1 = mypow( fabs(rCosPhi), alpha_minus_2 ) * rCosPhi;
                        mreal rCosPhiAlpha = rCosPhiAlphaMinus1 * rCosPhi;
                        
                        mreal Num = rCosPhiAlpha;
                        mreal factor0 = rBeta * alpha;
                        mreal density = rBeta * Num;
                        sum += a * b * density;
                        
                        mreal F = factor0 * rCosPhiAlphaMinus1;
                        mreal H = beta * rBetaMinus2 * Num;
                        
                        mreal bF = b * F;
                        
                        mreal Z1 = ( - n1 * F + v1 * H );
                        mreal Z2 = ( - n2 * F + v2 * H );
                        mreal Z3 = ( - n3 * F + v3 * H );
                        
                        P_U[ 7 * i + 0 ] += b * (
                                                 density
                                                 +
                                                 F * ( n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3) )
                                                 -
                                                 H * ( v1 * x1 + v2 * x2 + v3 * x3 )
                                                 );
                        P_U[ 7 * i + 1 ] += b  * Z1;
                        P_U[ 7 * i + 2 ] += b  * Z2;
                        P_U[ 7 * i + 3 ] += b  * Z3;
                        P_U[ 7 * i + 4 ] += bF * v1;
                        P_U[ 7 * i + 5 ] += bF * v2;
                        P_U[ 7 * i + 6 ] += bF * v3;
                        
                        C_U[ far_dim * C + 0 ] += a * (
                                                 density
                                                 -
                                                 F * ( n1 * y1 + n2 * y2 + n3 * y3 )
                                                 +
                                                 H * ( v1 * y1 + v2 * y2 + v3 * y3 )
                                                 );
                        C_U[ far_dim * C + 1 ] -= a  * Z1;
                        C_U[ far_dim * C + 2 ] -= a  * Z2;
                        C_U[ far_dim * C + 3 ] -= a  * Z3;
                    }
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
                            mreal x1 = P_X1[i];   mreal n1 = P_N1[i];
                            mreal x2 = P_X2[i];   mreal n2 = P_N2[i];
                            mreal x3 = P_X3[i];   mreal n3 = P_N3[i];
                            
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
                                    
                                    mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                                    mreal r2      = v1 * v1 + v2 * v2 + v3 * v3;
                                    
                                    mreal rBetaMinus2 = mypow( r2, minus_betahalf_minus_1 );
                                    mreal rBeta = rBetaMinus2 * r2;
                                    
                                    mreal rCosPhiAlphaMinus1 = mypow( fabs(rCosPhi), alpha_minus_2 ) * rCosPhi;
                                    mreal rCosPhiAlpha = rCosPhiAlphaMinus1 * rCosPhi;
                                    
                                    mreal Num = rCosPhiAlpha;
                                    mreal factor0 = rBeta * alpha;
                                    mreal density = rBeta * Num;
                                    sum += a * b * density;
                                    
                                    mreal F = factor0 * rCosPhiAlphaMinus1;
                                    mreal H = beta * rBetaMinus2 * Num;
                                    
                                    mreal bF = b * F;
                                    
                                    mreal Z1 = ( - n1 * F + v1 * H );
                                    mreal Z2 = ( - n2 * F + v2 * H );
                                    mreal Z3 = ( - n3 * F + v3 * H );
                                    
                                    P_U[ 7 * i + 0 ] += b * (
                                                             density
                                                             +
                                                             F * ( n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3) )
                                                             -
                                                             H * ( v1 * x1 + v2 * x2 + v3 * x3 )
                                                             );
                                    P_U[ 7 * i + 1 ] += b  * Z1;
                                    P_U[ 7 * i + 2 ] += b  * Z2;
                                    P_U[ 7 * i + 3 ] += b  * Z3;
                                    P_U[ 7 * i + 4 ] += bF * v1;
                                    P_U[ 7 * i + 5 ] += bF * v2;
                                    P_U[ 7 * i + 6 ] += bF * v3;
                                    
                                    P_U[ 7 * j + 0 ] += a * (
                                                             density
                                                             -
                                                             F * ( n1 * y1 + n2 * y2 + n3 * y3 )
                                                             +
                                                             H * ( v1 * y1 + v2 * y2 + v3 * y3 )
                                                             );
                                    P_U[ 7 * j + 1 ] -= a  * Z1;
                                    P_U[ 7 * j + 2 ] -= a  * Z2;
                                    P_U[ 7 * j + 3 ] -= a  * Z3;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        ptoc("TPEnergyBarnesHut0::DEnergy");
        return sum;
    }; // DEnergy
    
    double TPEnergyBarnesHut0::Value()
    {
        ptic("TPEnergyBarnesHut0::Value");
        
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
        ptoc("TPEnergyBarnesHut0::Value");
        
        return value;
    } // Value

    void TPEnergyBarnesHut0::Differential( Eigen::MatrixXd &output )
    {
        ptic("TPEnergyBarnesHut0::Differential");
        if( bvh->near_dim != 7)
        {
            eprint("in TPEnergyBarnesHut0::Differential: near_dim != 7");
            valprint("bvh->near_dim",bvh->near_dim);
        }
        
        EigenMatrixRM P_D_near( bvh->primitive_count, bvh->near_dim );
        EigenMatrixRM P_D_far ( bvh->primitive_count, bvh->far_dim );

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
        
        bvh->CollectDerivatives( P_D_near.data(), P_D_far.data() );
                
        AssembleDerivativeFromACNData( mesh, geom, P_D_near, output, weight );

        if( bvh->far_dim == 10)
        {
            AssembleDerivativeFromACPData( mesh, geom, P_D_far, output, weight );
        }
        else
        {
            AssembleDerivativeFromACNData( mesh, geom, P_D_far, output, weight );
        }
        ptoc("TPEnergyBarnesHut0::Differential");
    } // Differential
    
    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void TPEnergyBarnesHut0::Update()
    {
        ptic("TPEnergyBarnesHut0::Update");
        if (bvh)
        {
            delete bvh;
        }
        
        bvh = CreateOptimizedBVH(mesh, geom);
        ptoc("TPEnergyBarnesHut0::Update");
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 TPEnergyBarnesHut0::GetExponents()
    {
        return Vector2{alpha, beta};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *TPEnergyBarnesHut0::GetBVH()
    {
        return bvh;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double TPEnergyBarnesHut0::GetTheta()
    {
        return theta;
    }


} // namespace rsurfaces
