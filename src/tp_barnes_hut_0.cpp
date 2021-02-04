#include "block_cluster_tree2.h"

namespace rsurfaces
{

    //######################################################################################################################################
    //      Barnes-Hut Energy
    //######################################################################################################################################

    mreal BlockClusterTree2::BarnesHutEnergy()
    {
        
        mreal real_minus_betahalf = -beta/2.;
        mint ntreads = std::min( S->thread_count, T->thread_count);
        
        // Dunno why "restrict" helps with P_data. It is actually a lie here when S = T.
        mreal const * const restrict P_A  = S->P_data[0];
        mreal const * const restrict P_X1 = S->P_data[1];
        mreal const * const restrict P_X2 = S->P_data[2];
        mreal const * const restrict P_X3 = S->P_data[3];
        mreal const * const restrict P_N1 = S->P_data[4];
        mreal const * const restrict P_N2 = S->P_data[5];
        mreal const * const restrict P_N3 = S->P_data[6];

        mreal const * const restrict C_min1 = S->C_min[0];
        mreal const * const restrict C_min2 = S->C_min[1];
        mreal const * const restrict C_min3 = S->C_min[2];
        
        mreal const * const restrict C_max1 = S->C_max[0];
        mreal const * const restrict C_max2 = S->C_max[1];
        mreal const * const restrict C_max3 = S->C_max[2];
        
        mreal const * const restrict C_A  = S->C_data[0];
        mreal const * const restrict C_X1 = S->C_data[1];
        mreal const * const restrict C_X2 = S->C_data[2];
        mreal const * const restrict C_X3 = S->C_data[3];

        mint  const * const restrict C_left  = S->C_left;
        mint  const * const restrict C_right = S->C_right;
        mint  const * const restrict C_begin = S->C_begin;
        mint  const * const restrict C_end   = S->C_end;
        mreal const * const restrict C_r2    = S->C_squared_radius;
        
        mint  const * const restrict leaf = S->leaf_clusters;
        
        A_Vector<A_Vector<mint>> thread_stack ( thread_count );
        
        mreal sum = 0.;
        
        #pragma omp parallel for num_threads( ntreads ) reduction( + : sum)
        for( mint k = 0; k < S->leaf_cluster_count; ++k )
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
                        mreal x1 = P_X1[i];   mreal n1 = P_N1[i];
                        mreal x2 = P_X2[i];   mreal n2 = P_N2[i];
                        mreal x3 = P_X3[i];   mreal n3 = P_N3[i];
                        
                        mreal v1 = y1 - x1;
                        mreal v2 = y2 - x2;
                        mreal v3 = y3 - x3;
                        
                        mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                        mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 ;
                        local_local_sum += a * mypow( fabs(rCosPhi), alpha ) * mypow( r2, real_minus_betahalf );
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
                            mreal x1 = P_X1[i];   mreal n1 = P_N1[i];
                            mreal x2 = P_X2[i];   mreal n2 = P_N2[i];
                            mreal x3 = P_X3[i];   mreal n3 = P_N3[i];
                            
                            mreal local_local_sum = 0.;
                            
                            #pragma omp simd aligned( P_A, P_X1, P_X3 : ALIGN )
                            for( mint j = j_begin; j < j_end; ++j )
                            {
                                if ( i != j )
                                {
                                    mreal b  = P_A [j];
                                    mreal v1 = P_X1[j] - x1;
                                    mreal v2 = P_X2[j] - x2;
                                    mreal v3 = P_X3[j] - x3;
                                    
                                    mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
                                    mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 ;
                                    
                                    local_local_sum += mypow( fabs(rCosPhi), alpha ) * mypow( r2, real_minus_betahalf ) * b;
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
    }; //BarnesHutEnergyInteger


    mreal BlockClusterTree2::DBarnesHutEnergyHelper()
    {
        mreal real_alpha_minus_2 = alpha - 2.;
        mreal real_minus_betahalf_minus_1 = -beta/2.;

        
        mint data_dim = S->data_dim;
        mint ntreads = S->thread_count;
        
        // Dunno why "restrict" helps with P_data. It is actually a lie here when S = T.
        mreal const * const restrict P_A  = S->P_data[0];
        mreal const * const restrict P_X1 = S->P_data[1];
        mreal const * const restrict P_X2 = S->P_data[2];
        mreal const * const restrict P_X3 = S->P_data[3];
        mreal const * const restrict P_N1 = S->P_data[4];
        mreal const * const restrict P_N2 = S->P_data[5];
        mreal const * const restrict P_N3 = S->P_data[6];

        mreal const * const restrict C_min1 = S->C_min[0];
        mreal const * const restrict C_min2 = S->C_min[1];
        mreal const * const restrict C_min3 = S->C_min[2];
        
        mreal const * const restrict C_max1 = S->C_max[0];
        mreal const * const restrict C_max2 = S->C_max[1];
        mreal const * const restrict C_max3 = S->C_max[2];
        
        mreal const * const restrict C_A  = S->C_data[0];
        mreal const * const restrict C_X1 = S->C_data[1];
        mreal const * const restrict C_X2 = S->C_data[2];
        mreal const * const restrict C_X3 = S->C_data[3];

        
        mint  const * const restrict C_left  = S->C_left;
        mint  const * const restrict C_right = S->C_right;
        mint  const * const restrict C_begin = S->C_begin;
        mint  const * const restrict C_end   = S->C_end;
        mreal const * const restrict C_r2    = S->C_squared_radius;
        
        mint  const * const restrict leaf = S->leaf_clusters;
        
        A_Vector<A_Vector<mint>> thread_stack ( thread_count );
        
        mreal sum = 0.;
        
        #pragma omp parallel for num_threads( ntreads ) reduction( + : sum)
        for( mint k = 0; k < S->leaf_cluster_count; ++k )
        {
            mint thread = omp_get_thread_num();
            
            A_Vector<mint> * stack = &thread_stack[thread];
            
            mreal * const restrict P_U = &S->P_D[thread][0];
            mreal * const restrict C_U = &S->C_D[thread][0];
            
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
                        
                        mreal rBetaMinus2 = mypow( r2, real_minus_betahalf_minus_1 );
                        mreal rBeta = rBetaMinus2 * r2;
                        
                        mreal rCosPhiAlphaMinus1 = mypow( fabs(rCosPhi), real_alpha_minus_2 ) * rCosPhi;
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
                        
                        P_U[ data_dim * i + 0 ] += b * (
                                                        density
                                                        +
                                                        F * ( n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3) )
                                                        -
                                                        H * ( v1 * x1 + v2 * x2 + v3 * x3 )
                                                        );
                        C_U[ data_dim * C + 0 ] += a * (
                                                        density
                                                        -
                                                        F * ( n1 * y1 + n2 * y2 + n3 * y3 )
                                                        +
                                                        H * ( v1 * y1 + v2 * y2 + v3 * y3 )
                                                        );
                        
                        
                        P_U[ data_dim * i + 1 ] += b  * Z1;
                        P_U[ data_dim * i + 2 ] += b  * Z2;
                        P_U[ data_dim * i + 3 ] += b  * Z3;
                        P_U[ data_dim * i + 4 ] += bF * v1;
                        P_U[ data_dim * i + 5 ] += bF * v2;
                        P_U[ data_dim * i + 6 ] += bF * v3;
                        
                        C_U[ data_dim * C + 1 ] -= a  * Z1;
                        C_U[ data_dim * C + 2 ] -= a  * Z2;
                        C_U[ data_dim * C + 3 ] -= a  * Z3;
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
                                    
                                    mreal rBetaMinus2 = mypow( r2, real_minus_betahalf_minus_1 );
                                    mreal rBeta = rBetaMinus2 * r2;
                                    
                                    mreal rCosPhiAlphaMinus1 = mypow( fabs(rCosPhi), real_alpha_minus_2 ) * rCosPhi;
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
                                    
                                    P_U[ data_dim * i + 0 ] += b * (
                                                                    density
                                                                    +
                                                                    F * ( n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3) )
                                                                    -
                                                                    H * ( v1 * x1 + v2 * x2 + v3 * x3 )
                                                                    );
                                    P_U[ data_dim * j + 0 ] += a * (
                                                                    density
                                                                    -
                                                                    F * ( n1 * y1 + n2 * y2 + n3 * y3 )
                                                                    +
                                                                    H * ( v1 * y1 + v2 * y2 + v3 * y3 )
                                                                    );
                                    
                                    
                                    P_U[ data_dim * i + 1 ] += b  * Z1;
                                    P_U[ data_dim * i + 2 ] += b  * Z2;
                                    P_U[ data_dim * i + 3 ] += b  * Z3;
                                    P_U[ data_dim * i + 4 ] += bF * v1;
                                    P_U[ data_dim * i + 5 ] += bF * v2;
                                    P_U[ data_dim * i + 6 ] += bF * v3;
                                    
                                    P_U[ data_dim * j + 1 ] -= a  * Z1;
                                    P_U[ data_dim * j + 2 ] -= a  * Z2;
                                    P_U[ data_dim * j + 3 ] -= a  * Z3;
                                }
                            }
                        }
                    }
                }
            }
        }
        return sum;
    }; //DBarnesHutEnergyHelper

}
