#pragma once

#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <omp.h>
#include <mkl.h>
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

#define MKL_DIRECT_CALL_SEQ_JIT
#define EIGEN_NO_DEBUG

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
        std::string profile_path = ".";
        
        TreePercolationAlgorithm tree_perc_alg = TreePercolationAlgorithm::Tasks;
        MeshPtr mesh1;
        GeomPtr geom1;
        
        Eigen::MatrixXd U;
        Eigen::MatrixXd V;
        mreal E_11;
        Eigen::MatrixXd DE_11;
        
        void Compute( mint iter )
        {
            //        OptimizedClusterTree *bvh1 = CreateOptimizedBVH(mesh1, geom1);
            ptic("Energy");

            auto tpe_bh_11 = std::make_shared<TPEnergyBarnesHut0>(mesh1, geom1, alpha, beta, theta, weight);

            tpe_bh_11->GetBVH()->tree_perc_alg = tree_perc_alg;

//            if( iter < 0)
//            {
//                tpe_bh_11->GetBVH()->PrintToFile();
//            }

            E_11 = tpe_bh_11->Value();
            DE_11.setZero();
            tpe_bh_11->Differential(DE_11);
            ptoc("Energy");
            
            ptic("Multiply");
            auto bct11 = std::make_shared<OptimizedBlockClusterTree>(tpe_bh_11->GetBVH(), tpe_bh_11->GetBVH(), alpha, beta, chi);
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
            ptoc("Multiply");
        }
        
        void PrepareVectors()
        {
            ptic("PrepareVectors");
            mint vertex_count1 = mesh1->nVertices();
            DE_11 = Eigen::MatrixXd(vertex_count1, 3);
            
            U = getVertexPositions( mesh1, geom1 );
            V = Eigen::MatrixXd(vertex_count1, 3);
            V.setZero();
            ptoc("PrepareVectors");
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
    };
    
} // namespace rsurfaces
