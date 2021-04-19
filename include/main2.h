#pragma once

#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <omp.h>
#include <mkl.h>
#include <mkl_spblas.h>

#include <tbb/task_scheduler_init.h>
#include <memory>
#include <Eigen/Core>

#include "rsurface_types.h"
#include "surface_flow.h"


#include "remeshing/dynamic_remesher.h"
#include "remeshing/remeshing.h"

#include "scene_file.h"

#include "bct_kernel_type.h"
#include "optimized_bct.h"
#include "bct_constructors.h"

#include "helpers.h"


#include "energy/all_energies.h"


namespace rsurfaces
{
    
    
    struct Benchmarker
    {
        mint max_thread_count = 1;
        mint thread_count = 1;
        mint thread_step = 1;
        
        mint burn_ins = 1;
        mint iterations = 3;
        
        
        mreal alpha = 6.;
        mreal beta = 12.;
        mreal theta = 0.5;
        mreal chi = 0.5;
        mreal weight = 1.;
        
        std::string obj1 = "../scenes/Bunny/bunny.obj";
        std::string profile_name = "Profile";
        std::string path = ".";
        
        TreePercolationAlgorithm tree_perc_alg = TreePercolationAlgorithm::Tasks;
        MeshPtr mesh1;
        GeomPtr geom1;
        
        std::shared_ptr<TPPointCloudObstacleBarnesHut0> tp_o_pc;
        std::shared_ptr<TPPointNormalCloudObstacleBarnesHut0> tp_o_pnc;
        
        Eigen::MatrixXd U;
        Eigen::MatrixXd V;
        mreal E_11;
        Eigen::MatrixXd DE_11;
        
        void Compute( mint iter )
        {
            //        OptimizedClusterTree *bvh1 = CreateOptimizedBVH(mesh1, geom1);
            
            print("BVH");
            ptic("BVH");
            auto tpe_bh_11 = std::make_shared<TPEnergyBarnesHut0>(mesh1, geom1, alpha, beta, theta, weight);
            ptoc("BVH");

            tpe_bh_11->GetBVH()->tree_perc_alg = tree_perc_alg;
            
            if( iter < 0)
            {
                tpe_bh_11->GetBVH()->PrintToFile( path + "/" + "OptimizedClusterTree.tsv");
            }
            
            print("Energy");
            ptic("Energy");

            E_11 = tpe_bh_11->Value();
            DE_11.setZero();
            
            tpe_bh_11->Differential(DE_11);
            ptoc("Energy");

        
//            MeshUPtr u_pc_mesh;
//            GeomUPtr u_pc_geom;
//            std::tie(u_pc_mesh, u_pc_geom) = readMesh("../scenes/LungGrowing/sphere.obj");
//
//            mint n = u_pc_mesh->nVertices();
//            u_pc_geom->requireVertexNormals();
//
//            Eigen::MatrixXd pt_coords( n, 3);
//            Eigen::MatrixXd pt_normals( n, 3);
//            Eigen::VectorXd pt_weights( n);
//
//            #pragma omp parallel for
//            for( mint i = 0; i < n; ++i )
//            {
//                pt_weights(i) = 1.;
//                pt_coords(i, 0) = u_pc_geom->inputVertexPositions[i][0];
//                pt_coords(i, 1) = u_pc_geom->inputVertexPositions[i][1];
//                pt_coords(i, 2) = u_pc_geom->inputVertexPositions[i][2];
//                pt_normals(i, 0) = u_pc_geom->vertexNormals[i][0];
//                pt_normals(i, 1) = u_pc_geom->vertexNormals[i][1];
//                pt_normals(i, 2) = u_pc_geom->vertexNormals[i][2];
//            }
//
//
//            ptic("TPPointCloudObstacleBarnesHut0");
//
//            tp_o_pc = std::make_shared<TPPointCloudObstacleBarnesHut0>(mesh1, geom1, tpe_bh_11.get(), pt_weights, pt_coords, alpha, beta, theta, weight);
//
//            E_11 = tp_o_pc->Value();
//            DE_11.setZero();
//
//            tp_o_pc->Differential(DE_11);
//            ptoc("TPPointCloudObstacleBarnesHut0");
//
//            ptic("TPPointNormalCloundObstacleBarnesHut0");
//            tp_o_pnc = std::make_shared<TPPointNormalCloudObstacleBarnesHut0>(mesh1, geom1, tpe_bh_11.get(), pt_weights, pt_coords, pt_normals, alpha, beta, theta, weight);
//
//            E_11 = tp_o_pnc->Value();
//
//            DE_11.setZero();
//
//            tp_o_pnc->Differential(DE_11);
//
//            ptoc("TPPointNormalCloundObstacleBarnesHut0");
            
            
            std::shared_ptr<OptimizedBlockClusterTree> bct11;
            
            print("Multiply MKL CSR");
            ptic("Multiply MKL CSR");
            bct11 = std::make_shared<OptimizedBlockClusterTree>(tpe_bh_11->GetBVH(), tpe_bh_11->GetBVH(), alpha, beta, chi);
            bct11->mult_alg = NearFieldMultiplicationAlgorithm::MKL_CSR;
            for( mint k = 0; k < 20; ++k)
            {
                ptic("Multiply Fractional");
                bct11->Multiply(V,U,BCTKernelType::FractionalOnly);
                ptoc("Multiply Fractional");

                ptic("Multiply HighOrder");
                bct11->Multiply(V,U,BCTKernelType::HighOrder);
                ptoc("Multiply HighOrder");

                ptic("Multiply LowOrder");
                bct11->Multiply(V,U,BCTKernelType::LowOrder);
                ptoc("Multiply LowOrder");
            }
            ptoc("Multiply MKL CSR");
            

            print("Multiply Hybrid");
            ptic("Multiply Hybrid");
            bct11 = std::make_shared<OptimizedBlockClusterTree>(tpe_bh_11->GetBVH(), tpe_bh_11->GetBVH(), alpha, beta, chi);
            bct11->mult_alg = NearFieldMultiplicationAlgorithm::Hybrid;
            for( mint k = 0; k < 20; ++k)
            {
                ptic("Multiply Fractional");
                bct11->Multiply(V,U,BCTKernelType::FractionalOnly);
                ptoc("Multiply Fractional");

                ptic("Multiply HighOrder");
                bct11->Multiply(V,U,BCTKernelType::HighOrder);
                ptoc("Multiply HighOrder");

                ptic("Multiply LowOrder");
                bct11->Multiply(V,U,BCTKernelType::LowOrder);
                ptoc("Multiply LowOrder");
            }
            ptoc("Multiply Hybrid");
        }
        
        void Prepare()
        {
            ptic("Vectors");
            mint vertex_count1 = mesh1->nVertices();
            DE_11 = Eigen::MatrixXd(vertex_count1, 3);
            
            V = getVertexPositions( mesh1, geom1 );
            U = Eigen::MatrixXd(vertex_count1, 3);
            U.setZero();
            ptoc("Vectors");
            
        }
        
        void PrintStats()
        {
            std::cout << "mesh = "<< obj1 << std::endl;
            //    std::cout << "profile_file = "<< profile_file << std::endl;
            std::cout << "threads = "<< thread_count << std::endl;
            std::cout << "alpha = "<< alpha << std::endl;
            std::cout << "beta  = "<< beta << std::endl;
            std::cout << "theta = "<< theta << std::endl;
            std::cout << "chi   = "<< chi << std::endl;
        }
        
        void TestMultiply()
        {
            auto tpe_bh_11 = std::make_shared<TPEnergyBarnesHut0>(mesh1, geom1, alpha, beta, theta, weight);
            
            tpe_bh_11->GetBVH()->PrintToFile( path + "/" + "OptimizedClusterTree.tsv");
        
            
            //        E_11 = tpe_bh_11->Value();
            //        DE_11.setZero();
            //
            //        tpe_bh_11->Differential(DE_11);
            
            ptic("Multiply");
            auto bct11 = std::make_shared<OptimizedBlockClusterTree>(tpe_bh_11->GetBVH(), tpe_bh_11->GetBVH(), alpha, beta, chi);
            //            for( mint k = 0; k < 20; ++k)
            //            {
            //                ptic("Multiply Fractional");
            //                bct11->Multiply(V,U,BCTKernelType::FractionalOnly);
            //                ptoc("Multiply Fractional");
            //
            //                ptic("Multiply HighOrder");
            //                bct11->Multiply(V,U,BCTKernelType::HighOrder);
            //                ptoc("Multiply HighOrder");
            //
            //                ptic("Multiply LowOrder");
            //                bct11->Multiply(V,U,BCTKernelType::LowOrder);
            //                ptoc("Multiply LowOrder");
            //            }
            
            
            OptimizedClusterTree * S = bct11->S;
            OptimizedClusterTree * T = bct11->T;
            S->RequireBuffers(1);
            T->RequireBuffers(1);
            
            S->CleanseBuffers();
            T->CleanseBuffers();
            
            mint n = T->cluster_count;
            
            Eigen::VectorXd v ( n );
            Eigen::VectorXd u0 ( n );
            Eigen::VectorXd u1 ( n );
            Eigen::VectorXd u2 ( n );
            
            
            std::uniform_real_distribution<double> unif(-1.,1.);
            std::default_random_engine re;
            
            for( mint i = 0; i < n; ++i)
            {
                v(i) = unif(re);
            }
            
            T->RequireChunks();
            
            print("\n################################");
            print("PercolateUp");
            
            T->CleanseBuffers();
            std::copy( v.data(), v.data() + n, T->C_in );
            T->tree_perc_alg = TreePercolationAlgorithm::Tasks;
            T->PercolateUp();
            std::copy( T->C_in, T->C_in + n, u0.data() );
            
            T->CleanseBuffers();
            std::copy( v.data(), v.data() + n, T->C_in );
            T->tree_perc_alg = TreePercolationAlgorithm::Sequential;
            T->PercolateUp();
            std::copy( T->C_in, T->C_in + n, u1.data() );
            
            T->CleanseBuffers();
            std::copy( v.data(), v.data() +n, T->C_in );
            T->tree_perc_alg = TreePercolationAlgorithm::Chunks;
            T->PercolateUp();
            std::copy( T->C_in, T->C_in + n, u2.data() );
            
            valprint("u0.norm()", u0.norm() );
            
            valprint("Absolute error Tasks-Sequential", (u0-u1).norm() );
            valprint("Relative error Tasks-Sequential", (u0-u1).norm()/u0.norm());
            
            valprint("Absolute error Tasks-Chunks", (u2-u0).norm() );
            valprint("Relative error Tasks-Chunks", (u2-u0).norm()/u0.norm() );
            
            valprint("Absolute error Sequential-Chunks", (u2-u1).norm() );
            valprint("Relative error Sequential-Chunks", (u2-u1).norm()/u1.norm() );
            
            print("\n################################");
            print("PercolateDown");
            
            tic("Tasks");
            T->CleanseBuffers();
            std::copy( v.data(), v.data() + n, T->C_out );
            T->tree_perc_alg = TreePercolationAlgorithm::Tasks;
            T->PercolateDown();
            std::copy( T->C_out, T->C_out + n, u0.data() );
            toc("Tasks");
            
            tic("Sequential");
            T->CleanseBuffers();
            std::copy( v.data(), v.data() + n, T->C_out );
            T->tree_perc_alg = TreePercolationAlgorithm::Sequential;
            T->PercolateDown();
            std::copy( T->C_out, T->C_out + n, u1.data() );
            toc("Sequential");
            
            tic("Chunks");
            T->CleanseBuffers();
            std::copy( v.data(), v.data() + n, T->C_out );
            T->tree_perc_alg = TreePercolationAlgorithm::Chunks;
            T->PercolateDown();
            std::copy( T->C_out, T->C_out + n, u2.data() );
            toc("Chunks");
            
            valprint("u0.norm()", u0.norm() );
            
            valprint("Absolute error Tasks-Sequential", (u0-u1).norm() );
            valprint("Relative error Tasks-Sequential", (u0-u1).norm()/u0.norm());
            
            valprint("Absolute error Tasks-Chunks", (u2-u0).norm() );
            valprint("Relative error Tasks-Chunks", (u2-u0).norm()/u0.norm() );
            
            valprint("Absolute error Sequential-Chunks", (u2-u1).norm() );
            valprint("Relative error Sequential-Chunks", (u2-u1).norm()/u1.norm() );
            
            

            print("\n################################");
            print("Full Multiply");
                
            BCTKernelType type = BCTKernelType::LowOrder;
            mint m = T->primitive_count;
            mint cols = 3;
            
            S->RequireBuffers(cols);
            T->RequireBuffers(cols);
            
            Eigen::MatrixXd V ( m , cols );
            Eigen::MatrixXd V0 ( m , cols );
            
            for( mint i = 0; i < m; ++i)
            {
                for( mint j = 0; j < cols; ++ j)
                {
                    V0(i,j) = V(i,j) = unif(re);
                }
            }
            
            //            T->Pre( V, type );
            
            Eigen::MatrixXd U0 ( m , cols );
            U0.setZero();
            T->CleanseBuffers();
            S->CleanseBuffers();
            bct11->S->tree_perc_alg = TreePercolationAlgorithm::Tasks;
            bct11->Multiply(V,U0,type, false);
            
            Eigen::MatrixXd U1 ( m , cols );
            U1.setZero();
            T->CleanseBuffers();
            S->CleanseBuffers();
            bct11->S->tree_perc_alg = TreePercolationAlgorithm::Sequential;
            bct11->Multiply(V,U1,type, false);
            
            Eigen::MatrixXd U2 ( m , cols );
            U2.setZero();
            T->CleanseBuffers();
            S->CleanseBuffers();
            bct11->S->tree_perc_alg = TreePercolationAlgorithm::Chunks;
            bct11->Multiply(V,U2,type, false);

            valprint("(V-V0).norm()", (V-V0).norm() );
            
            valprint("Absolute multiplication error Tasks-Sequential", (U0-U1).norm() );
            valprint("Relative multiplication error Tasks-Sequential", (U0-U1).norm()/U0.norm() );
                     
            valprint("Absolute multiplication error Tasks-Chunks", (U2-U0).norm() );
            valprint("Relative multiplication error Tasks-Chunks", (U2-U0).norm()/U0.norm() );
                
            valprint("Absolute multiplication error Sequential-Chunks", (U2-U1).norm() );
            valprint("Relative multiplication error Sequential-Chunks", (U2-U1).norm()/U1.norm() );
            
            
            ptoc("Multiply");
        }
        
//        void TestMKLOptimize()
//        {
//            auto tpe_bh_11 = std::make_shared<TPEnergyBarnesHut0>(mesh1, geom1, alpha, beta, theta, weight);
//
//
//            auto bct11 = std::make_shared<OptimizedBlockClusterTree>(tpe_bh_11->GetBVH(), tpe_bh_11->GetBVH(), alpha, beta, chi);
//
//
//
//            mint m = bct11->near->m;
//            mint n = bct11->near->n;
//            mint cols = 9;
//            mreal * values = bct11->near->hi_values;
//            mreal factor = 1.;
//
//
//            Eigen::VectorXd v ( n * cols );
//            std::uniform_real_distribution<double> unif(-1.,1.);
//            std::default_random_engine re;
//
//            for( mint i = 0; i < n * cols; ++i)
//            {
//                v(i) = unif(re);
//            }
//
//            Eigen::VectorXd u1 ( n * cols );
//            Eigen::VectorXd u2 ( n * cols );
//
//            sparse_matrix_t A = NULL;
//            sparse_status_t stat = mkl_sparse_d_create_csr ( &A, SPARSE_INDEX_BASE_ZERO, m, n, bct11->near->OuterPtrB(), bct11->near->OuterPtrE(), bct11->near->InnerPtr(), values );
//            if (stat)
//            {
//                eprint("mkl_sparse_d_create_csr returned stat = " + std::to_string(stat) );
//            }
//
//            mint repetitions = 20;
//
//            tic("MKL_CSR unoptimized");
//            for( mint i = 0; i < repetitions; ++i )
//            {
//                stat = mkl_sparse_d_mm ( SPARSE_OPERATION_NON_TRANSPOSE, factor, A, bct11->near->descr, SPARSE_LAYOUT_ROW_MAJOR, v.data(), cols, cols, 0., u1.data(), cols );
//                if (stat)
//                {
//                    eprint("mkl_sparse_d_mm returned stat = " + std::to_string(stat) );
//                }
//            }
//            toc("MKL_CSR unoptimized");
//
//            tic("optimization");
//            stat = mkl_sparse_set_mm_hint(A, SPARSE_OPERATION_NON_TRANSPOSE, bct11->near->descr, SPARSE_LAYOUT_ROW_MAJOR, cols, 100 * repetitions);
//            if (stat)
//            {
//                eprint("mkl_sparse_set_mm_hint returned stat = " + std::to_string(stat) );
//            }
//
//            stat = mkl_sparse_optimize( A );
//            if (stat)
//            {
//                eprint("mkl_sparse_optimize = " + std::to_string(stat) );
//            }
//            toc("optimization");
//
//            tic("MKL_CSR optimized");
//            for( mint i = 0; i < repetitions; ++i )
//            {
//                stat = mkl_sparse_d_mm ( SPARSE_OPERATION_NON_TRANSPOSE, factor, A, bct11->near->descr, SPARSE_LAYOUT_ROW_MAJOR, v.data(), cols, cols, 0., u2.data(), cols );
//                if (stat)
//                {
//                    eprint("mkl_sparse_d_mm returned stat = " + std::to_string(stat) );
//                }
//            }
//            toc("MKL_CSR optimized");
//
//
//            valprint("(u1-u2).norm()",(u1-u2).norm());
//            valprint("(u1-u2).norm()/u1.norm()",(u1-u2).norm()/u1.norm());
//
//        }
        
//        void TestMKLOptimize()
//        {
//            auto tpe_bh_11 = std::make_shared<TPEnergyBarnesHut0>(mesh1, geom1, alpha, beta, theta, weight);
//
//
//            auto bct11 = std::make_shared<OptimizedBlockClusterTree>(tpe_bh_11->GetBVH(), tpe_bh_11->GetBVH(), alpha, beta, chi);
//
//
//            MKLSparseMatrix matrix = bct11->S->P_to_C;
//
//            mint m = matrix.m;
//            mint n = matrix.n;
//            mint cols = 9;
//
//            mreal factor = 1.;
//
//
//            Eigen::VectorXd v ( n * cols );
//            std::uniform_real_distribution<double> unif(-1.,1.);
//            std::default_random_engine re;
//
//            for( mint i = 0; i < n * cols; ++i)
//            {
//                v(i) = unif(re);
//            }
//
//            Eigen::VectorXd u1 ( m * cols );
//            Eigen::VectorXd u2 ( m * cols );
//
//
//
//            sparse_matrix_t A = NULL;
//            sparse_status_t stat = mkl_sparse_d_create_csr ( &A, SPARSE_INDEX_BASE_ZERO, m, n, matrix.outer, matrix.outer + 1 , matrix.inner, matrix.values );
//            if (stat)
//            {
//                eprint("mkl_sparse_d_create_csr returned stat = " + std::to_string(stat) );
//            }
//
//            mint repetitions = 20;
//
//            tic("MKL_CSR unoptimized");
//            for( mint i = 0; i < repetitions; ++i )
//            {
//                stat = mkl_sparse_d_mm ( SPARSE_OPERATION_NON_TRANSPOSE, factor, A, matrix.descr, SPARSE_LAYOUT_ROW_MAJOR, v.data(), cols, cols, 0., u1.data(), cols );
//                if (stat)
//                {
//                    eprint("mkl_sparse_d_mm returned stat = " + std::to_string(stat) );
//                }
//            }
//            toc("MKL_CSR unoptimized");
//
//            tic("optimization");
//            stat = mkl_sparse_set_mm_hint(A, SPARSE_OPERATION_NON_TRANSPOSE, matrix.descr, SPARSE_LAYOUT_ROW_MAJOR, cols, 100 * repetitions);
//            if (stat)
//            {
//                eprint("mkl_sparse_set_mm_hint returned stat = " + std::to_string(stat) );
//            }
//
//            stat = mkl_sparse_optimize( A );
//            if (stat)
//            {
//                eprint("mkl_sparse_optimize = " + std::to_string(stat) );
//            }
//            toc("optimization");
//
//            tic("MKL_CSR optimized");
//            for( mint i = 0; i < repetitions; ++i )
//            {
//                stat = mkl_sparse_d_mm ( SPARSE_OPERATION_NON_TRANSPOSE, factor, A, matrix.descr, SPARSE_LAYOUT_ROW_MAJOR, v.data(), cols, cols, 0., u2.data(), cols );
//                if (stat)
//                {
//                    eprint("mkl_sparse_d_mm returned stat = " + std::to_string(stat) );
//                }
//            }
//            toc("MKL_CSR optimized");
//
//
//            valprint("(u1-u2).norm()",(u1-u2).norm());
//            valprint("(u1-u2).norm()/u1.norm()",(u1-u2).norm()/u1.norm());
//
//        }

        
//        void TestVBSR()
//        {
//            
//            omp_set_num_threads(1);
//            mkl_set_num_threads(1);
//            
//            auto tpe = std::make_shared<TPEnergyBarnesHut0>(mesh1, geom1, alpha, beta, theta, weight);
//
//            auto bct = std::make_shared<OptimizedBlockClusterTree>(tpe->GetBVH(), tpe->GetBVH(), alpha, beta, chi);
//
//            mint n = bct->near->n;
//            mint m = bct->near->m;
//            mint cols = 9;
//            
//            Eigen::VectorXd v  ( n * cols );
//            Eigen::VectorXd u1 ( m * cols );
//            Eigen::VectorXd u2 ( m * cols );
//            Eigen::VectorXd u3 ( m * cols );
//            
//            std::uniform_real_distribution<double> unif(-1.,1.);
//            std::default_random_engine re;
//            for( mint i = 0; i < n * cols; ++i)
//            {
//                v(i) = unif(re);
//            }
//            
//            tic("ApplyKernel_CSR_MKL");
//            for( mint i = 0; i < 20; ++i)
//            {
//                bct->near->ApplyKernel_CSR_MKL( bct->near->hi_values, v.data(), u1.data(), cols, 1. );
//            }
//            toc("ApplyKernel_CSR_MKL");
//            
//            tic("ApplyKernel_VBSR");
//            for( mint i = 0; i < 20; ++i)
//            {
//                bct->near->ApplyKernel_VBSR( bct->near->hi_values, v.data(), u2.data(), cols, 1. );
//            }
//            toc("ApplyKernel_VBSR");
//            
//            tic("ApplyKernel_Hybrid");
//            for( mint i = 0; i < 20; ++i)
//            {
//                bct->near->ApplyKernel_Hybrid( bct->near->hi_values, v.data(), u3.data(), cols, 1. );
//            }
//            toc("ApplyKernel_Hybrid");
//            
//            
//            
//            omp_set_num_threads(4);
//            mkl_set_num_threads(4);
//            
//            tpe = std::make_shared<TPEnergyBarnesHut0>(mesh1, geom1, alpha, beta, theta, weight);
//
//            bct = std::make_shared<OptimizedBlockClusterTree>(tpe->GetBVH(), tpe->GetBVH(), alpha, beta, chi);
//            
//            for( mint i = 0; i < n * cols; ++i)
//            {
//                v(i) = unif(re);
//            }
//            
//            tic("ApplyKernel_CSR_MKL");
//            for( mint i = 0; i < 20; ++i)
//            {
//                bct->near->ApplyKernel_CSR_MKL( bct->near->hi_values, v.data(), u1.data(), cols, 1. );
//            }
//            toc("ApplyKernel_CSR_MKL");
//            
//            tic("ApplyKernel_VBSR");
//            for( mint i = 0; i < 20; ++i)
//            {
//                bct->near->ApplyKernel_VBSR( bct->near->hi_values, v.data(), u2.data(), cols, 1. );
//            }
//            toc("ApplyKernel_VBSR");
//            
//            tic("ApplyKernel_Hybrid");
//            for( mint i = 0; i < 20; ++i)
//            {
//                bct->near->ApplyKernel_Hybrid( bct->near->hi_values, v.data(), u3.data(), cols, 1. );
//            }
//            toc("ApplyKernel_Hybrid");
//            
//            
//            omp_set_num_threads(max_thread_count);
//            mkl_set_num_threads(max_thread_count);
//            
//            tpe = std::make_shared<TPEnergyBarnesHut0>(mesh1, geom1, alpha, beta, theta, weight);
//
//            bct = std::make_shared<OptimizedBlockClusterTree>(tpe->GetBVH(), tpe->GetBVH(), alpha, beta, chi);
//            
//            for( mint i = 0; i < n * cols; ++i)
//            {
//                v(i) = unif(re);
//            }
//            
//            tic("ApplyKernel_CSR_MKL");
//            for( mint i = 0; i < 20; ++i)
//            {
//                bct->near->ApplyKernel_CSR_MKL( bct->near->hi_values, v.data(), u1.data(), cols, 1. );
//            }
//            toc("ApplyKernel_CSR_MKL");
//            
//            tic("ApplyKernel_VBSR");
//            for( mint i = 0; i < 20; ++i)
//            {
//                bct->near->ApplyKernel_VBSR( bct->near->hi_values, v.data(), u2.data(), cols, 1. );
//            }
//            toc("ApplyKernel_VBSR");
//            
//            tic("ApplyKernel_Hybrid");
//            for( mint i = 0; i < 20; ++i)
//            {
//                bct->near->ApplyKernel_Hybrid( bct->near->hi_values, v.data(), u3.data(), cols, 1. );
//            }
//            toc("ApplyKernel_Hybrid");
//            
////            valprint("(u1-u2).norm()/u1.norm()",(u1-u2).norm()/u1.norm());
////            valprint("(u1-u3).norm()/u1.norm()",(u1-u2).norm()/u1.norm());
//        }
        
    }; // Benchmarker
} // namespace rsurfaces
