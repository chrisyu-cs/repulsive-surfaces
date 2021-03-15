
#include "energy/tp_obstacle_barnes_hut_pr_0.h"

namespace rsurfaces
{
    double TPObstacleBarnesHut0::Value()
    {
        bvh = bvhSharedFrom->GetBVH();
        if (!bvh)
        {
            throw std::runtime_error("Obstacle energy is sharing BVH from an energy that has no BVH.");
        }
        if (use_int)
        {
            mint int_alpha = std::round(alpha);
            mint int_betahalf = std::round(beta / 2);
            return weight * Energy(int_alpha, int_betahalf);
        }
        else
        {
            mreal real_alpha = alpha;
            mreal real_betahalf = beta / 2;
            return weight * Energy(real_alpha, real_betahalf);
        }
    } // Value

    void TPObstacleBarnesHut0::Differential(Eigen::MatrixXd &output)
    {
        bvh = bvhSharedFrom->GetBVH();
        if (!bvh)
        {
            throw std::runtime_error("Obstacle energy is sharing BVH from an energy that has no BVH.");
        }
        
        if( bvh->data_dim != 7)
        {
            eprint("in TPObstacleAllPairs::Differential: data_dim != 7");
        }
        
        EigenMatrixRM P_D_data ( bvh->primitive_count, 7 );
        
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
    
        AssembleDerivativeFromACNData( mesh, geom, P_D_data, output, weight );
  
    } // Differential
    
    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void TPObstacleBarnesHut0::Update()
    {
        // Invalidate the old BVH pointer
        bvh = 0;
        // bvhSharedFrom is responsible for reallocating it in its Update() function
        bvh = bvhSharedFrom->GetBVH();
        if (!bvh)
        {
            throw std::runtime_error("Obstacle energy is sharing BVH from an energy that has no BVH.");
        }
    }

    // Get the mesh associated with this energy.
    MeshPtr TPObstacleBarnesHut0::GetMesh()
    {
        return mesh;
    }

    // Get the geometry associated with this geometry.
    GeomPtr TPObstacleBarnesHut0::GetGeom()
    {
        return geom;
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

//    void TPObstacleBarnesHut0::Differential(Eigen::MatrixXd &output)
//    {
//        EigenMatrixRM P_D_data(mesh->nFaces(), 7);
//
//        bvh->CleanseD();
//
//        if (use_int)
//        {
//            mint int_alpha = std::round(alpha);
//            mint int_betahalf = std::round(beta / 2);
//            DEnergy(int_alpha, int_betahalf);
//        }
//        else
//        {
//            mreal real_alpha = alpha;
//            mreal real_betahalf = beta / 2;
//            DEnergy(real_alpha, real_betahalf);
//        }
//
//        bvh->CollectDerivatives(P_D_data.data());
//
//        mint vertex_count = mesh->nVertices();
//        VertexIndices vInds = mesh->getVertexIndices();
//        FaceIndices fInds = mesh->getFaceIndices();
//
//        geom->requireVertexDualAreas();
//        geom->requireFaceAreas();
//        geom->requireCotanLaplacian();
//        geom->requireVertexPositions();
//
//        Eigen::MatrixXd buffer(mesh->nFaces() * 3, 3);
//
//        for (auto face : mesh->faces())
//        {
//            mint i = fInds[face];
//
//            GCHalfedge he = face.halfedge();
//
//            mint i0 = vInds[he.vertex()];
//            mint i1 = vInds[he.next().vertex()];
//            mint i2 = vInds[he.next().next().vertex()];
//
//            mreal x00 = geom->inputVertexPositions[i0][0];
//            mreal x01 = geom->inputVertexPositions[i0][1];
//            mreal x02 = geom->inputVertexPositions[i0][2];
//
//            mreal x10 = geom->inputVertexPositions[i1][0];
//            mreal x11 = geom->inputVertexPositions[i1][1];
//            mreal x12 = geom->inputVertexPositions[i1][2];
//
//            mreal x20 = geom->inputVertexPositions[i2][0];
//            mreal x21 = geom->inputVertexPositions[i2][1];
//            mreal x22 = geom->inputVertexPositions[i2][2];
//
//            mreal s0 = x11;
//            mreal s1 = x20;
//            mreal s2 = x10;
//            mreal s3 = x21;
//            mreal s4 = -s0;
//            mreal s5 = s3 + s4;
//            mreal s6 = -s1;
//            mreal s7 = s2 + s6;
//            mreal s8 = x12;
//            mreal s9 = x22;
//            mreal s10 = x00;
//            mreal s11 = -s8;
//            mreal s12 = s11 + s9;
//            mreal s13 = x01;
//            mreal s14 = s13 * s7;
//            mreal s15 = s0 * s1;
//            mreal s16 = -(s2 * s3);
//            mreal s17 = s10 * s5;
//            mreal s18 = s14 + s15 + s16 + s17;
//            mreal s19 = s18 * s18;
//            mreal s20 = x02;
//            mreal s21 = s20 * s7;
//            mreal s22 = s1 * s8;
//            mreal s23 = -(s2 * s9);
//            mreal s24 = s10 * s12;
//            mreal s25 = s21 + s22 + s23 + s24;
//            mreal s26 = s25 * s25;
//            mreal s27 = -s3;
//            mreal s28 = s0 + s27;
//            mreal s29 = s20 * s28;
//            mreal s30 = s3 * s8;
//            mreal s31 = -(s0 * s9);
//            mreal s32 = s12 * s13;
//            mreal s33 = s29 + s30 + s31 + s32;
//            mreal s34 = s33 * s33;
//            mreal s35 = s19 + s26 + s34;
//            mreal s36 = sqrt(s35);
//            mreal s37 = 1 / s36;
//            mreal s38 = 2 * s18 * s5;
//            mreal s39 = 2 * s12 * s25;
//            mreal s40 = s38 + s39;
//            mreal s41 = P_D_data(i, 0);
//            mreal s42 = s1 + s10 + s2;
//            mreal s43 = 2 * s18 * s7;
//            mreal s44 = 2 * s12 * s33;
//            mreal s45 = s43 + s44;
//            mreal s46 = P_D_data(i, 1);
//            mreal s47 = s0 + s13 + s3;
//            mreal s48 = s36 / 6.;
//            mreal s49 = P_D_data(i, 2);
//            mreal s50 = s20 + s8 + s9;
//            mreal s51 = P_D_data(i, 3);
//            mreal s52 = P_D_data(i, 6);
//            mreal s53 = 2 * s25 * s7;
//            mreal s54 = 2 * s28 * s33;
//            mreal s55 = s53 + s54;
//            mreal s56 = P_D_data(i, 4);
//            mreal s57 = P_D_data(i, 5);
//            mreal s58 = -s9;
//            mreal s59 = s13 + s27;
//            mreal s60 = 2 * s18 * s59;
//            mreal s61 = s20 + s58;
//            mreal s62 = 2 * s25 * s61;
//            mreal s63 = s60 + s62;
//            mreal s64 = -s10;
//            mreal s65 = s1 + s64;
//            mreal s66 = 2 * s18 * s65;
//            mreal s67 = 2 * s33 * s61;
//            mreal s68 = s66 + s67;
//            mreal s69 = -s13;
//            mreal s70 = s3 + s69;
//            mreal s71 = 2 * s25 * s65;
//            mreal s72 = 2 * s33 * s70;
//            mreal s73 = s71 + s72;
//            mreal s74 = -s20;
//            mreal s75 = s0 + s69;
//            mreal s76 = 2 * s18 * s75;
//            mreal s77 = s74 + s8;
//            mreal s78 = 2 * s25 * s77;
//            mreal s79 = s76 + s78;
//            mreal s80 = -s2;
//            mreal s81 = s10 + s80;
//            mreal s82 = 2 * s18 * s81;
//            mreal s83 = 2 * s33 * s77;
//            mreal s84 = s82 + s83;
//            mreal s85 = s13 + s4;
//            mreal s86 = 2 * s25 * s81;
//            mreal s87 = 2 * s33 * s85;
//            mreal s88 = s86 + s87;
//            buffer(3 * i + 0, 0) = (s37 * s40 * s41) / 4. + s46 * ((s37 * s40 * s42) / 12. + s48) + (s37 * s40 * s47 * s49) / 12. + (s37 * s40 * s50 * s51) / 12. + (s28 * s52) / 2. + (s12 * s57) / 2.;
//            buffer(3 * i + 0, 1) = (s37 * s41 * s45) / 4. + (s37 * s42 * s45 * s46) / 12. + ((s37 * s45 * s47) / 12. + s48) * s49 + (s37 * s45 * s50 * s51) / 12. + (s56 * (s58 + s8)) / 2. + (s52 * (s1 + s80)) / 2.;
//            buffer(3 * i + 0, 2) = (s37 * s41 * s55) / 4. + (s37 * s42 * s46 * s55) / 12. + (s37 * s47 * s49 * s55) / 12. + s51 * (s48 + (s37 * s50 * s55) / 12.) + (s5 * s56) / 2. + (s57 * s7) / 2.;
//            buffer(3 * i + 1, 0) = (s57 * s61) / 2. + (s37 * s41 * s63) / 4. + (s37 * s47 * s49 * s63) / 12. + (s37 * s50 * s51 * s63) / 12. + s46 * (s48 + (s37 * s42 * s63) / 12.) + (s52 * s70) / 2.;
//            buffer(3 * i + 1, 1) = (s52 * (s10 + s6)) / 2. + (s37 * s41 * s68) / 4. + (s37 * s42 * s46 * s68) / 12. + (s37 * s50 * s51 * s68) / 12. + s49 * (s48 + (s37 * s47 * s68) / 12.) + (s56 * (s74 + s9)) / 2.;
//            buffer(3 * i + 1, 2) = (s56 * s59) / 2. + (s57 * s65) / 2. + (s37 * s41 * s73) / 4. + (s37 * s42 * s46 * s73) / 12. + (s37 * s47 * s49 * s73) / 12. + s51 * (s48 + (s37 * s50 * s73) / 12.);
//            buffer(3 * i + 2, 0) = (s57 * s77) / 2. + (s37 * s41 * s79) / 4. + (s37 * s47 * s49 * s79) / 12. + (s37 * s50 * s51 * s79) / 12. + s46 * (s48 + (s37 * s42 * s79) / 12.) + (s52 * s85) / 2.;
//            buffer(3 * i + 2, 1) = ((s11 + s20) * s56) / 2. + (s52 * (s2 + s64)) / 2. + (s37 * s41 * s84) / 4. + (s37 * s42 * s46 * s84) / 12. + (s37 * s50 * s51 * s84) / 12. + s49 * (s48 + (s37 * s47 * s84) / 12.);
//            buffer(3 * i + 2, 2) = (s56 * s75) / 2. + (s57 * s81) / 2. + (s37 * s41 * s88) / 4. + (s37 * s42 * s46 * s88) / 12. + (s37 * s47 * s49 * s88) / 12. + s51 * (s48 + (s37 * s50 * s88) / 12.);
//        }
//
//        output += weight * (DerivativeAssembler(mesh, geom) * buffer);
//
//    } // Differential

    template <typename T1, typename T2>
    mreal TPObstacleBarnesHut0::Energy(T1 alpha, T2 betahalf)
    {
        T2 minus_betahalf = -betahalf;
        mreal theta2 = theta * theta;

        mint nthreads = bvh->thread_count;

        mreal sum = 0.;

        {
            auto S = bvh;
            auto T = o_bvh;
            mreal const *const restrict C_xmin1 = S->C_min[0];
            mreal const *const restrict C_xmin2 = S->C_min[1];
            mreal const *const restrict C_xmin3 = S->C_min[2];
            mreal const *const restrict C_xmax1 = S->C_max[0];
            mreal const *const restrict C_xmax2 = S->C_max[1];
            mreal const *const restrict C_xmax3 = S->C_max[2];

            mreal const *const restrict C_xr2 = S->C_squared_radius;

            mreal const *const restrict P_A = S->P_data[0];
            mreal const *const restrict P_X1 = S->P_data[1];
            mreal const *const restrict P_X2 = S->P_data[2];
            mreal const *const restrict P_X3 = S->P_data[3];
            mreal const *const restrict P_N1 = S->P_data[4];
            mreal const *const restrict P_N2 = S->P_data[5];
            mreal const *const restrict P_N3 = S->P_data[6];

            mint const *const restrict C_xbegin = S->C_begin;
            mint const *const restrict C_xend = S->C_end;

            mint const *const restrict leaf = S->leaf_clusters;

            mreal const *const restrict C_ymin1 = T->C_min[0];
            mreal const *const restrict C_ymin2 = T->C_min[1];
            mreal const *const restrict C_ymin3 = T->C_min[2];
            mreal const *const restrict C_ymax1 = T->C_max[0];
            mreal const *const restrict C_ymax2 = T->C_max[1];
            mreal const *const restrict C_ymax3 = T->C_max[2];

            mreal const *const restrict C_yr2 = T->C_squared_radius;

            mreal const *const restrict P_B = T->P_data[0];
            mreal const *const restrict P_Y1 = T->P_data[1];
            mreal const *const restrict P_Y2 = T->P_data[2];
            mreal const *const restrict P_Y3 = T->P_data[3];

            mreal const *const restrict C_B = T->C_data[0];
            mreal const *const restrict C_Y1 = T->C_data[1];
            mreal const *const restrict C_Y2 = T->C_data[2];
            mreal const *const restrict C_Y3 = T->C_data[3];

            mint const *const restrict C_ybegin = T->C_begin;
            mint const *const restrict C_yend = T->C_end;

            mint const *const restrict C_left = T->C_left;
            mint const *const restrict C_right = T->C_right;

            A_Vector<A_Vector<mint>> thread_stack(nthreads);

#pragma omp parallel for num_threads(nthreads) reduction(+ \
                                                         : sum)
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

                    // Compute squared distance between bounding boxes.
                    // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares

                    mreal ymin1 = C_ymin1[C];
                    mreal ymin2 = C_ymin2[C];
                    mreal ymin3 = C_ymin3[C];

                    mreal ymax1 = C_ymax1[C];
                    mreal ymax2 = C_ymax2[C];
                    mreal ymax3 = C_ymax3[C];

                    mreal d1 = mymax(0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1));
                    mreal d2 = mymax(0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2));
                    mreal d3 = mymax(0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3));

                    mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;

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

        {
            auto S = o_bvh;
            auto T = bvh;
            mreal const *const restrict C_xmin1 = S->C_min[0];
            mreal const *const restrict C_xmin2 = S->C_min[1];
            mreal const *const restrict C_xmin3 = S->C_min[2];
            mreal const *const restrict C_xmax1 = S->C_max[0];
            mreal const *const restrict C_xmax2 = S->C_max[1];
            mreal const *const restrict C_xmax3 = S->C_max[2];

            mreal const *const restrict C_xr2 = S->C_squared_radius;

            mreal const *const restrict P_A = S->P_data[0];
            mreal const *const restrict P_X1 = S->P_data[1];
            mreal const *const restrict P_X2 = S->P_data[2];
            mreal const *const restrict P_X3 = S->P_data[3];
            mreal const *const restrict P_N1 = S->P_data[4];
            mreal const *const restrict P_N2 = S->P_data[5];
            mreal const *const restrict P_N3 = S->P_data[6];

            mint const *const restrict C_xbegin = S->C_begin;
            mint const *const restrict C_xend = S->C_end;

            mint const *const restrict leaf = S->leaf_clusters;

            mreal const *const restrict C_ymin1 = T->C_min[0];
            mreal const *const restrict C_ymin2 = T->C_min[1];
            mreal const *const restrict C_ymin3 = T->C_min[2];
            mreal const *const restrict C_ymax1 = T->C_max[0];
            mreal const *const restrict C_ymax2 = T->C_max[1];
            mreal const *const restrict C_ymax3 = T->C_max[2];

            mreal const *const restrict C_yr2 = T->C_squared_radius;

            mreal const *const restrict P_B = T->P_data[0];
            mreal const *const restrict P_Y1 = T->P_data[1];
            mreal const *const restrict P_Y2 = T->P_data[2];
            mreal const *const restrict P_Y3 = T->P_data[3];

            mreal const *const restrict C_B = T->C_data[0];
            mreal const *const restrict C_Y1 = T->C_data[1];
            mreal const *const restrict C_Y2 = T->C_data[2];
            mreal const *const restrict C_Y3 = T->C_data[3];

            mint const *const restrict C_ybegin = T->C_begin;
            mint const *const restrict C_yend = T->C_end;

            mint const *const restrict C_left = T->C_left;
            mint const *const restrict C_right = T->C_right;

            A_Vector<A_Vector<mint>> thread_stack(nthreads);

#pragma omp parallel for num_threads(nthreads) reduction(+ \
                                                         : sum)
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

                    // Compute squared distance between bounding boxes.
                    // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares

                    mreal ymin1 = C_ymin1[C];
                    mreal ymin2 = C_ymin2[C];
                    mreal ymin3 = C_ymin3[C];

                    mreal ymax1 = C_ymax1[C];
                    mreal ymax2 = C_ymax2[C];
                    mreal ymax3 = C_ymax3[C];

                    mreal d1 = mymax(0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1));
                    mreal d2 = mymax(0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2));
                    mreal d3 = mymax(0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3));

                    mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;

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

        return sum;
    }; //Energy

    template <typename T1, typename T2>
    mreal TPObstacleBarnesHut0::DEnergy(T1 alpha, T2 betahalf)
    {

        T1 alpha_minus_2 = alpha - 2;
        T2 minus_betahalf_minus_1 = -betahalf - 1;

        mreal beta = 2. * betahalf;
        mreal theta2 = theta * theta;
        mreal sum = 0.;

        mint data_dim = bvh->data_dim;
        mint nthreads = bvh->thread_count;

        {
            auto S = bvh;
            auto T = o_bvh;
            mreal const *const restrict C_xmin1 = S->C_min[0];
            mreal const *const restrict C_xmin2 = S->C_min[1];
            mreal const *const restrict C_xmin3 = S->C_min[2];
            mreal const *const restrict C_xmax1 = S->C_max[0];
            mreal const *const restrict C_xmax2 = S->C_max[1];
            mreal const *const restrict C_xmax3 = S->C_max[2];

            mreal const *const restrict C_xr2 = S->C_squared_radius;

            mreal const *const restrict P_A = S->P_data[0];
            mreal const *const restrict P_X1 = S->P_data[1];
            mreal const *const restrict P_X2 = S->P_data[2];
            mreal const *const restrict P_X3 = S->P_data[3];
            mreal const *const restrict P_N1 = S->P_data[4];
            mreal const *const restrict P_N2 = S->P_data[5];
            mreal const *const restrict P_N3 = S->P_data[6];

            mint const *const restrict C_xbegin = S->C_begin;
            mint const *const restrict C_xend = S->C_end;

            mint const *const restrict leaf = S->leaf_clusters;

            mreal const *const restrict C_ymin1 = T->C_min[0];
            mreal const *const restrict C_ymin2 = T->C_min[1];
            mreal const *const restrict C_ymin3 = T->C_min[2];
            mreal const *const restrict C_ymax1 = T->C_max[0];
            mreal const *const restrict C_ymax2 = T->C_max[1];
            mreal const *const restrict C_ymax3 = T->C_max[2];

            mreal const *const restrict C_yr2 = T->C_squared_radius;

            mreal const *const restrict P_B = T->P_data[0];
            mreal const *const restrict P_Y1 = T->P_data[1];
            mreal const *const restrict P_Y2 = T->P_data[2];
            mreal const *const restrict P_Y3 = T->P_data[3];

            mreal const *const restrict C_B = T->C_data[0];
            mreal const *const restrict C_Y1 = T->C_data[1];
            mreal const *const restrict C_Y2 = T->C_data[2];
            mreal const *const restrict C_Y3 = T->C_data[3];

            mint const *const restrict C_ybegin = T->C_begin;
            mint const *const restrict C_yend = T->C_end;

            mint const *const restrict C_left = T->C_left;
            mint const *const restrict C_right = T->C_right;

            A_Vector<A_Vector<mint>> thread_stack(nthreads);

#pragma omp parallel for num_threads(nthreads) reduction(+ \
                                                         : sum)
            for (mint k = 0; k < S->leaf_cluster_count; ++k)
            {
                mint thread = omp_get_thread_num();

                A_Vector<mint> *stack = &thread_stack[thread];

                mreal *const restrict P_U = &S->P_D_data[thread][0];
                //            mreal * const restrict P_V = &T->P_D_data[thread][0];
                //            mreal * const restrict C_V = &T->C_D_data[thread][0];

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

                    // Compute squared distance between bounding boxes.
                    // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares

                    mreal ymin1 = C_ymin1[C];
                    mreal ymin2 = C_ymin2[C];
                    mreal ymin3 = C_ymin3[C];

                    mreal ymax1 = C_ymax1[C];
                    mreal ymax2 = C_ymax2[C];
                    mreal ymax3 = C_ymax3[C];

                    mreal d1 = mymax(0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1));
                    mreal d2 = mymax(0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2));
                    mreal d3 = mymax(0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3));

                    mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;

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

                            P_U[data_dim * i] += b * (density +
                                                      F * (n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3)) -
                                                      H * (v1 * x1 + v2 * x2 + v3 * x3));
                            P_U[data_dim * i + 1] += b * Z1;
                            P_U[data_dim * i + 2] += b * Z2;
                            P_U[data_dim * i + 3] += b * Z3;
                            P_U[data_dim * i + 4] += bF * v1;
                            P_U[data_dim * i + 5] += bF * v2;
                            P_U[data_dim * i + 6] += bF * v3;

                            //                        C_V[ 7 * C + 0 ] += a * (
                            //                                                 density
                            //                                                 -
                            //                                                 F * ( n1 * y1 + n2 * y2 + n3 * y3 )
                            //                                                 +
                            //                                                 H * ( v1 * y1 + v2 * y2 + v3 * y3 )
                            //                                                 );
                            //                        C_V[ 7 * C + 1 ] -= a  * Z1;
                            //                        C_V[ 7 * C + 2 ] -= a  * Z2;
                            //                        C_V[ 7 * C + 3 ] -= a  * Z3;
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

#pragma omp simd aligned(P_B, P_Y1, P_Y2, P_Y3 \
                         : ALIGN) reduction(+  \
                                            : sum)
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

                                    //                                P_V[ 7 * j + 0 ] += a * (
                                    //                                                         density
                                    //                                                         -
                                    //                                                         F * ( n1 * y1 + n2 * y2 + n3 * y3 )
                                    //                                                         +
                                    //                                                         H * ( v1 * y1 + v2 * y2 + v3 * y3 )
                                    //                                                         );
                                    //                                P_V[ 7 * j + 1 ] -= a  * Z1;
                                    //                                P_V[ 7 * j + 2 ] -= a  * Z2;
                                    //                                P_V[ 7 * j + 3 ] -= a  * Z3;
                                }

                                P_U[data_dim * i] += da;
                                P_U[data_dim * i + 1] += dx1;
                                P_U[data_dim * i + 2] += dx2;
                                P_U[data_dim * i + 3] += dx3;
                                P_U[data_dim * i + 4] += dn1;
                                P_U[data_dim * i + 5] += dn2;
                                P_U[data_dim * i + 6] += dn3;
                            }
                        }
                    }
                }
            }
        }

        {
            auto S = o_bvh;
            auto T = bvh;
            mreal const *const restrict C_xmin1 = S->C_min[0];
            mreal const *const restrict C_xmin2 = S->C_min[1];
            mreal const *const restrict C_xmin3 = S->C_min[2];
            mreal const *const restrict C_xmax1 = S->C_max[0];
            mreal const *const restrict C_xmax2 = S->C_max[1];
            mreal const *const restrict C_xmax3 = S->C_max[2];

            mreal const *const restrict C_xr2 = S->C_squared_radius;

            mreal const *const restrict P_A = S->P_data[0];
            mreal const *const restrict P_X1 = S->P_data[1];
            mreal const *const restrict P_X2 = S->P_data[2];
            mreal const *const restrict P_X3 = S->P_data[3];
            mreal const *const restrict P_N1 = S->P_data[4];
            mreal const *const restrict P_N2 = S->P_data[5];
            mreal const *const restrict P_N3 = S->P_data[6];

            mint const *const restrict C_xbegin = S->C_begin;
            mint const *const restrict C_xend = S->C_end;

            mint const *const restrict leaf = S->leaf_clusters;

            mreal const *const restrict C_ymin1 = T->C_min[0];
            mreal const *const restrict C_ymin2 = T->C_min[1];
            mreal const *const restrict C_ymin3 = T->C_min[2];
            mreal const *const restrict C_ymax1 = T->C_max[0];
            mreal const *const restrict C_ymax2 = T->C_max[1];
            mreal const *const restrict C_ymax3 = T->C_max[2];

            mreal const *const restrict C_yr2 = T->C_squared_radius;

            mreal const *const restrict P_B = T->P_data[0];
            mreal const *const restrict P_Y1 = T->P_data[1];
            mreal const *const restrict P_Y2 = T->P_data[2];
            mreal const *const restrict P_Y3 = T->P_data[3];

            mreal const *const restrict C_B = T->C_data[0];
            mreal const *const restrict C_Y1 = T->C_data[1];
            mreal const *const restrict C_Y2 = T->C_data[2];
            mreal const *const restrict C_Y3 = T->C_data[3];

            mint const *const restrict C_ybegin = T->C_begin;
            mint const *const restrict C_yend = T->C_end;

            mint const *const restrict C_left = T->C_left;
            mint const *const restrict C_right = T->C_right;

            A_Vector<A_Vector<mint>> thread_stack(nthreads);

#pragma omp parallel for num_threads(nthreads) reduction(+ \
                                                         : sum)
            for (mint k = 0; k < S->leaf_cluster_count; ++k)
            {
                mint thread = omp_get_thread_num();

                A_Vector<mint> *stack = &thread_stack[thread];

                //            mreal * const restrict P_U = &S->P_D_data[thread][0];
                mreal *const restrict P_V = &T->P_D_data[thread][0];
                mreal *const restrict C_V = &T->C_D_data[thread][0];

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

                    // Compute squared distance between bounding boxes.
                    // Inpired by https://gamedev.stackexchange.com/questions/154036/efficient-minimum-distance-between-two-axis-aligned-squares

                    mreal ymin1 = C_ymin1[C];
                    mreal ymin2 = C_ymin2[C];
                    mreal ymin3 = C_ymin3[C];

                    mreal ymax1 = C_ymax1[C];
                    mreal ymax2 = C_ymax2[C];
                    mreal ymax3 = C_ymax3[C];

                    mreal d1 = mymax(0., mymax(xmin1, ymin1) - mymin(xmax1, ymax1));
                    mreal d2 = mymax(0., mymax(xmin2, ymin2) - mymin(xmax2, ymax2));
                    mreal d3 = mymax(0., mymax(xmin3, ymin3) - mymin(xmax3, ymax3));

                    mreal R2 = d1 * d1 + d2 * d2 + d3 * d3;

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

#pragma omp simd aligned(P_A, P_X1, P_X2, P_X3, P_N1, P_N2, P_N3 \
                         : ALIGN) reduction(+                    \
                                            : sum)
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

                            //                        P_U[ data_dim * i     ] += b * (
                            //                                   density
                            //                                   +
                            //                                   F * ( n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3) )
                            //                                   -
                            //                                   H * ( v1 * x1 + v2 * x2 + v3 * x3 )
                            //                                   );
                            //                        P_U[ data_dim * i + 1 ] += b  * Z1;
                            //                        P_U[ data_dim * i + 2 ] += b  * Z2;
                            //                        P_U[ data_dim * i + 3 ] += b  * Z3;
                            //                        P_U[ data_dim * i + 4 ] += bF * v1;
                            //                        P_U[ data_dim * i + 5 ] += bF * v2;
                            //                        P_U[ data_dim * i + 6 ] += bF * v3;

                            db += a * (density -
                                       F * (n1 * y1 + n2 * y2 + n3 * y3) +
                                       H * (v1 * y1 + v2 * y2 + v3 * y3));
                            dy1 -= a * Z1;
                            dy2 -= a * Z2;
                            dy3 -= a * Z3;
                        }
                        C_V[7 * C + 0] += db;
                        C_V[7 * C + 1] += dy1;
                        C_V[7 * C + 2] += dy2;
                        C_V[7 * C + 3] += dy3;
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

                                //                            mreal  da = 0.;
                                //                            mreal dx1 = 0.;
                                //                            mreal dx2 = 0.;
                                //                            mreal dx3 = 0.;
                                //                            mreal dn1 = 0.;
                                //                            mreal dn2 = 0.;
                                //                            mreal dn3 = 0.;

#pragma omp simd aligned(P_B, P_Y1, P_Y2, P_Y3, P_V \
                         : ALIGN) reduction(+       \
                                            : sum)
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

                                    //                                da += b * (
                                    //                                           density
                                    //                                           +
                                    //                                           F * ( n1 * (x1 - v1) + n2 * (x2 - v2) + n3 * (x3 - v3) )
                                    //                                           -
                                    //                                           H * ( v1 * x1 + v2 * x2 + v3 * x3 )
                                    //                                           );
                                    //                                dx1 += b  * Z1;
                                    //                                dx2 += b  * Z2;
                                    //                                dx3 += b  * Z3;
                                    //                                dn1 += bF * v1;
                                    //                                dn2 += bF * v2;
                                    //                                dn3 += bF * v3;

                                    P_V[7 * j + 0] += a * (density -
                                                           F * (n1 * y1 + n2 * y2 + n3 * y3) +
                                                           H * (v1 * y1 + v2 * y2 + v3 * y3));
                                    P_V[7 * j + 1] -= a * Z1;
                                    P_V[7 * j + 2] -= a * Z2;
                                    P_V[7 * j + 3] -= a * Z3;
                                }
                                //                            P_U[ data_dim * i     ] +=  da;
                                //                            P_U[ data_dim * i + 1 ] += dx1;
                                //                            P_U[ data_dim * i + 2 ] += dx2;
                                //                            P_U[ data_dim * i + 3 ] += dx3;
                                //                            P_U[ data_dim * i + 4 ] += dn1;
                                //                            P_U[ data_dim * i + 5 ] += dn2;
                                //                            P_U[ data_dim * i + 6 ] += dn3;
                            }
                        }
                    }
                }
            }
        }

        return sum;
    }; // DEnergy
} // namespace rsurfaces
