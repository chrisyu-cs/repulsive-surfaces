#include "main2.h"

using namespace rsurfaces;
using namespace geometrycentral;
using namespace geometrycentral::surface;

struct Benchmarker
{
    mint max_thread_count = 1;
    mint thread_count = 1;
    mint thread_steps = 1;

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
    MeshPtr mesh1;
    GeomPtr geom1;
    
    void Compute()
    {
        mint vertex_count1 = mesh1->nVertices();
        
        OptimizedClusterTree *bvh1 = CreateOptimizedBVH(mesh1, geom1);
        auto bct11 = std::make_shared<OptimizedBlockClusterTree>(bvh1, bvh1, alpha, beta, chi);
        
//        bct11->PrintStats();
        
        auto tpe_bh_11 = std::make_shared<TPEnergyBarnesHut0>(mesh1, geom1, alpha, beta, theta, weight);
        auto tpe_fm_11 = std::make_shared<TPEnergyMultipole0>(mesh1, geom1, bct11.get(), alpha, beta, weight);
        
        mreal E_11;
        Eigen::MatrixXd DE_11(vertex_count1, 3);
        
        E_11 = tpe_bh_11->Value();
        DE_11.setZero();
        tpe_bh_11->Differential(DE_11);
        
        E_11 = tpe_fm_11->Value();
        DE_11.setZero();
        tpe_fm_11->Differential(DE_11);
        
        Eigen::MatrixXd U(vertex_count1, 3);
        U = getVertexPositions( mesh1, geom1 );
        
        Eigen::MatrixXd V(vertex_count1, 3);
        V.setZero();
        
        for( mint k = 0; k < 15; ++k)
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

        delete bvh1;
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

int main(int arg_count, char* arg_vec[])
{
    using namespace rsurfaces;
    
    namespace po = boost::program_options;
    
    auto BM = Benchmarker();
    
    #pragma omp parallel
    {
        BM.thread_count = BM.max_thread_count = omp_get_num_threads()/2;
    }

    
    po::options_description desc("Allowed options");
    
    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("mesh", po::value<std::string>(), "file of mesh to use as variable")
            ("profile_path", po::value<std::string>(), "path to store the profile")
            ("profile_name", po::value<std::string>(), "file base name of profile file")
            ("alpha", po::value<mreal>(), "file name of mesh to use as variable")
            ("beta", po::value<mreal>(), "file name of mesh to use as variable")
            ("theta", po::value<mreal>(), "separation parameter for barnes-hut method")
            ("chi", po::value<mreal>(), "separation parameter for block cluster tree")
            ("threads", po::value<mint>(), "number of threads to be used")
            ("thread_steps", po::value<mint>(), "number of threads to be used")
        
            ("burn_ins", po::value<mint>(), "number of burn-in iterations to use")
            ("iterations", po::value<mint>(), "number of iterations to use for the benchmark")
        ;

        po::variables_map var_map;
        po::store(po::parse_command_line(arg_count, arg_vec, desc), var_map);
        po::notify(var_map);

        if (var_map.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

        if (var_map.count("mesh")) {
            BM.obj1 = var_map["mesh"].as<std::string>();
        }
        if (var_map.count("profile_name")) {
            BM.profile_name = var_map["profile_name"].as<std::string>();
        }
        if (var_map.count("profile_path")) {
            BM.profile_path = var_map["profile_path"].as<std::string>();
        }

        if (var_map.count("theta")) {
            BM.theta = var_map["theta"].as<mreal>();
        }
        if (var_map.count("chi")) {
            BM.chi = var_map["chi"].as<mreal>();
        }
        if (var_map.count("alpha")) {
            BM.alpha = var_map["alpha"].as<mreal>();
        }
        if (var_map.count("beta")) {
            BM.beta = var_map["beta"].as<mreal>();
        }

        if (var_map.count("threads")) {
            BM.thread_count = var_map["threads"].as<mint>();
            BM.max_thread_count = var_map["threads"].as<mint>();
        }
        if (var_map.count("thread_steps")) {
            BM.thread_steps = var_map["thread_steps"].as<mint>();
        }
        if (var_map.count("burn_ins")) {
            BM.burn_ins = var_map["burn_ins"].as<mint>();
        }
        if (var_map.count("iterations")) {
            BM.iterations = var_map["iterations"].as<mint>();
        }
    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }


    
    
    std::cout << std::setprecision(8);

    MeshUPtr u_mesh;
    GeomUPtr u_geom;

    // Load mesh1
    std::cout << "Loading " << BM.obj1 << " as variable." << std::endl;
    std::tie(u_mesh, u_geom) = readMesh(BM.obj1);
    BM.mesh1 = std::move(u_mesh);
    BM.geom1 = std::move(u_geom);
    BM.geom1->requireVertexDualAreas();
    BM.geom1->requireVertexNormals();
    
//    // Load mesh2
//    std::string obj2 = "../scenes/LungGrowing/sphere.obj";
//    std::cout << "Loading " << obj2 << " as obstacle." << std::endl;
//    std::tie(u_mesh, u_geom) = readMesh(obj2);
//    MeshPtr mesh2 = std::move(u_mesh);
//    GeomPtr geom2 = std::move(u_geom);
//    geom2->requireVertexDualAreas();
//    geom2->requireVertexNormals();
//    mint primitive_count2 = mesh2->nVertices();

    for( mint threads = 0; threads < BM.max_thread_count + 1; threads += BM.thread_steps )
    {
    
        BM.thread_count = threads;
        omp_set_num_threads(BM.thread_count);

        std::cout << std::endl;
        std::cout << "### threads =  " << BM.thread_count << std::endl;
        
        ClearProfile(BM.profile_path + "/" + BM.profile_name + "_" + std::to_string(BM.thread_count) + ".tsv");
        
        //burn-in
        for( mint i = 0; i < BM.burn_ins; ++i)
        {
            std::cout << "burn_in " << i+1 << " / " << BM.burn_ins << std::endl;
            BM.Compute();

        }
        
        ClearProfile(BM.profile_path + "/" + BM.profile_name + "_" + std::to_string(BM.thread_count) + ".tsv");
        
        //the actual test code
        for( mint i = 0; i < BM.iterations; ++i)
        {
            std::cout << "iterations " << i+1 << " / " << BM.iterations << std::endl;
            ptic("Iteration");
            BM.Compute();
            ptoc("Iteration");

        }
        std::cout << std::endl;
    }
    
//    tic("Create bvh1");
//    OptimizedClusterTree *bvh1 = CreateOptimizedBVH(mesh1, geom1);
//    OptimizedClusterTree *bvh1_nl = CreateOptimizedBVH_Normals(mesh1, geom1);
//    OptimizedClusterTree *bvh1_pr = CreateOptimizedBVH_Projectors(mesh1, geom1);
//    toc("Create bvh1");
//    tic("Create bvh2");
//    OptimizedClusterTree *bvh2 = CreateOptimizedBVH(mesh2, geom2);
//    OptimizedClusterTree *bvh2_nl = CreateOptimizedBVH_Normals(mesh2, geom2);
//    OptimizedClusterTree *bvh2_pr = CreateOptimizedBVH_Projectors(mesh2, geom2);
//    toc("Create bvh2");
//
//    tic("Create bct11");
//    auto bct11 = std::make_shared<OptimizedBlockClusterTree>(bvh1, bvh1, alpha, beta, theta);
//    auto bct11_nl = std::make_shared<OptimizedBlockClusterTree>(bvh1_nl, bvh1_nl, alpha, beta, theta);
//    auto bct11_pr = std::make_shared<OptimizedBlockClusterTree>(bvh1_pr, bvh1_pr, alpha, beta, theta);
//    toc("Create bct11");
//    tic("Create bct12");
//    auto bct12 = std::make_shared<OptimizedBlockClusterTree>(bvh1, bvh2, alpha, beta, theta);
//    auto bct12_nl = std::make_shared<OptimizedBlockClusterTree>(bvh1_nl, bvh2_nl, alpha, beta, theta);
//    auto bct12_pr = std::make_shared<OptimizedBlockClusterTree>(bvh1_pr, bvh2_pr, alpha, beta, theta);
//    toc("Create bct12");
//
//    // The transpose of bct12 and thus not needed.
//    //auto bct21 = std::make_shared<OptimizedBlockClusterTree>(bvh2, bvh1, alpha, beta, theta);
//    tic("Create bct22");
//    auto bct22 = std::make_shared<OptimizedBlockClusterTree>(bvh2, bvh2, alpha, beta, theta);
//    auto bct22_nl = std::make_shared<OptimizedBlockClusterTree>(bvh2_nl, bvh2_nl, alpha, beta, theta);
//    auto bct22_pr = std::make_shared<OptimizedBlockClusterTree>(bvh2_pr, bvh2_pr, alpha, beta, theta);
//    toc("Create bct22");
//
//    bct11->PrintStats();
//    bct12->PrintStats();
//    bct22->PrintStats();
//
//    // The joint bct of the union of mesh1 and mesh2 can be written in block matrix for as
//    //  bct = {
//    //            { bct11, bct12 },
//    //            { bct21, bct22 }
//    //        },
//    // where bct11 and bct22 are the instances of OptimizedBlockClusterTree of mesh1 and mesh2, respectively, bct12 is cross interaction OptimizedBlockClusterTree of mesh1 and mesh2, and bct21 is the transpose of bct12.
//    // However, the according matrix (on the space of dofs on the primitives) would be
//    //  A   = {
//    //            { A11 + diag( A12 * one2 ) , A12                      },
//    //            { A21                      , A22 + diag( A21 * one1 ) }
//    //        },
//    // where one1 and one2 are all-1-vectors on the primitives of mesh1 and mesh2, respectively.
//    // OptimizedBlockClusterTree::AddObstacleCorrection is supposed to compute diag( A12 * one2 ) and to add it to the diagonal of A11.
//    // Afterwards, bct1->Multiply will also multiply with the metric contribution of the obstacle.
//    tic("Modifying bct11 to include the terms with respect to the obstacle.");
//    bct11->AddObstacleCorrection(bct12.get());
//    bct11_nl->AddObstacleCorrection(bct12_nl.get());
//    bct11_pr->AddObstacleCorrection(bct12_pr.get());
//    toc("Modifying bct11 to include the terms with respect to the obstacle.");
//
//    mint energy_count = 7;
//
//    // the self-interaction energy of mesh1
//    auto tpe_fm_11 = std::make_shared<TPEnergyMultipole0>(mesh1, geom1, bct11.get(), alpha, beta, weight);
//    auto tpe_fm_nl_11 = std::make_shared<TPEnergyMultipole_Normals0>(mesh1, geom1, bct11_nl.get(), alpha, beta, weight);
//    auto tpe_fm_pr_11 = std::make_shared<TPEnergyMultipole_Projectors0>(mesh1, geom1, bct11_pr.get(), alpha, beta, weight);
//    auto tpe_bh_11 = std::make_shared<TPEnergyBarnesHut0>(mesh1, geom1, alpha, beta, theta, weight);
//    auto tpe_bh_pr_11 = std::make_shared<TPEnergyBarnesHut_Projectors0>(mesh1, geom1, alpha, beta, theta, weight);
//    auto tpe_ex_11 = std::make_shared<TPEnergyAllPairs>(mesh1, geom1, alpha, beta, weight);
//    auto tpe_ex_pr_11 = std::make_shared<TPEnergyAllPairs_Projectors>(mesh1, geom1, alpha, beta, weight);
//
//    // the interaction energy between mesh1 and mesh2
//    auto tpe_fm_12 = std::make_shared<TPObstacleMultipole0>(mesh1, geom1, bct12.get(), alpha, beta, weight);
//    auto tpe_fm_nl_12 = std::make_shared<TPObstacleMultipole_Normals0>(mesh1, geom1, bct12_nl.get(), alpha, beta, weight);
//    auto tpe_fm_pr_12 = std::make_shared<TPObstacleMultipole_Projectors0>(mesh1, geom1, bct12_pr.get(), alpha, beta, weight);
//    auto tpe_bh_12 = std::make_shared<TPObstacleBarnesHut0>(mesh1, geom1, tpe_bh_11.get(), mesh2, geom2, alpha, beta, theta, weight);
//    auto tpe_bh_pr_12 = std::make_shared<TPObstacleBarnesHut_Projectors0>(mesh1, geom1, tpe_bh_pr_11.get(), mesh2, geom2, alpha, beta, theta, weight);
//    auto tpe_ex_12 = std::make_shared<TPObstacleAllPairs>(mesh1, geom1, tpe_bh_11.get(), mesh2, geom2, alpha, beta, weight);
//    auto tpe_ex_pr_12 = std::make_shared<TPObstacleAllPairs_Projectors>(mesh1, geom1, tpe_bh_pr_11.get(), mesh2, geom2, alpha, beta, weight);
//
//    // the self-interaction energy of mesh2; since mesh2 is the obstacle here, this is not needed in practice; I used this here only for test purposes and in order to see how much "work" is saved by this approach.
//    auto tpe_fm_22 = std::make_shared<TPEnergyMultipole0>(mesh2, geom2, bct22.get(), alpha, beta, weight);
//    auto tpe_fm_nl_22 = std::make_shared<TPEnergyMultipole_Normals0>(mesh2, geom2, bct22_nl.get(), alpha, beta, weight);
//    auto tpe_fm_pr_22 = std::make_shared<TPEnergyMultipole_Projectors0>(mesh2, geom2, bct22_pr.get(), alpha, beta, weight);
//    auto tpe_bh_22 = std::make_shared<TPEnergyBarnesHut0>(mesh2, geom2, alpha, beta, theta, weight);
//    auto tpe_bh_pr_22 = std::make_shared<TPEnergyBarnesHut_Projectors0>(mesh2, geom2, alpha, beta, theta, weight);
//    auto tpe_ex_22 = std::make_shared<TPEnergyAllPairs>(mesh2, geom2, alpha, beta, weight);
//    auto tpe_ex_pr_22 = std::make_shared<TPEnergyAllPairs_Projectors>(mesh2, geom2, alpha, beta, weight);
//
//    // the energies tpe_**_11, tpe_**_12, tpe_**_22 are gauged such that their sum equals the tangent-point energy of the union of mesh1 and mesh2.
//
//    double E_fm_11, E_fm_12, E_fm_22;
//    double E_fm_nl_11, E_fm_nl_12, E_fm_nl_22;
//    double E_fm_pr_11, E_fm_pr_12, E_fm_pr_22;
//    double E_bh_11, E_bh_12, E_bh_22;
//    double E_bh_pr_11, E_bh_pr_12, E_bh_pr_22;
//    double E_ex_11, E_ex_12, E_ex_22;
//    double E_ex_pr_11, E_ex_pr_12, E_ex_pr_22;
//
//    Eigen::MatrixXd DE_fm_11(primitive_count1, 3);
//    Eigen::MatrixXd DE_fm_12(primitive_count1, 3);
//    Eigen::MatrixXd DE_fm_22(primitive_count2, 3);
//
//    Eigen::MatrixXd DE_fm_nl_11(primitive_count1, 3);
//    Eigen::MatrixXd DE_fm_nl_12(primitive_count1, 3);
//    Eigen::MatrixXd DE_fm_nl_22(primitive_count2, 3);
//
//    Eigen::MatrixXd DE_fm_pr_11(primitive_count1, 3);
//    Eigen::MatrixXd DE_fm_pr_12(primitive_count1, 3);
//    Eigen::MatrixXd DE_fm_pr_22(primitive_count2, 3);
//
//    Eigen::MatrixXd DE_bh_11(primitive_count1, 3);
//    Eigen::MatrixXd DE_bh_12(primitive_count1, 3);
//    Eigen::MatrixXd DE_bh_22(primitive_count2, 3);
//
//    Eigen::MatrixXd DE_bh_pr_11(primitive_count1, 3);
//    Eigen::MatrixXd DE_bh_pr_12(primitive_count1, 3);
//    Eigen::MatrixXd DE_bh_pr_22(primitive_count2, 3);
//
//    Eigen::MatrixXd DE_ex_11(primitive_count1, 3);
//    Eigen::MatrixXd DE_ex_12(primitive_count1, 3);
//    Eigen::MatrixXd DE_ex_22(primitive_count2, 3);
//
//    Eigen::MatrixXd DE_ex_pr_11(primitive_count1, 3);
//    Eigen::MatrixXd DE_ex_pr_12(primitive_count1, 3);
//    Eigen::MatrixXd DE_ex_pr_22(primitive_count2, 3);
//
//    std::cout << "Using integer exponents." << std::endl;
//
//    mint counter = 0;
//    mint count = energy_count * 6;
//    tic();
//    E_ex_11 = tpe_ex_11->Value();
//    mreal t_ex_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    E_ex_12 = tpe_ex_12->Value();
//    mreal t_ex_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    E_ex_22 = tpe_ex_22->Value();
//    mreal t_ex_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_ex_11.setZero();
//    tpe_ex_11->Differential(DE_ex_11);
//    mreal Dt_ex_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_ex_12.setZero();
//    tpe_ex_12->Differential(DE_ex_12);
//    mreal Dt_ex_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_ex_22.setZero();
//    tpe_ex_22->Differential(DE_ex_22);
//    mreal Dt_ex_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    //######################################
//
//    tic();
//    E_ex_pr_11 = tpe_ex_pr_11->Value();
//    mreal t_ex_pr_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
////        tpe_ex_pr_12->Update();
//    E_ex_pr_12 = tpe_ex_pr_12->Value();
//    mreal t_ex_pr_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    E_ex_pr_22 = tpe_ex_pr_22->Value();
//    mreal t_ex_pr_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_ex_pr_11.setZero();
//    tpe_ex_pr_11->Differential(DE_ex_pr_11);
//    mreal Dt_ex_pr_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_ex_pr_12.setZero();
//    tpe_ex_pr_12->Differential(DE_ex_pr_12);
//    mreal Dt_ex_pr_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_ex_pr_22.setZero();
//    tpe_ex_pr_22->Differential(DE_ex_pr_22);
//    mreal Dt_ex_pr_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    //######################################
//
//
//    tic();
//    E_bh_11 = tpe_bh_11->Value();
//    mreal t_bh_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
////        tpe_bh_12->Update();
//    E_bh_12 = tpe_bh_12->Value();
//    mreal t_bh_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    E_bh_22 = tpe_bh_22->Value();
//    mreal t_bh_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_bh_11.setZero();
//    tpe_bh_11->Differential(DE_bh_11);
//    mreal Dt_bh_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_bh_12.setZero();
//    tpe_bh_12->Differential(DE_bh_12);
//    mreal Dt_bh_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_bh_22.setZero();
//    tpe_bh_22->Differential(DE_bh_22);
//    mreal Dt_bh_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    //######################################
//
//
//    tic();
//    E_bh_pr_11 = tpe_bh_pr_11->Value();
//    mreal t_bh_pr_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
////        tpe_bh_12->Update();
//    E_bh_pr_12 = tpe_bh_pr_12->Value();
//    mreal t_bh_pr_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    E_bh_pr_22 = tpe_bh_pr_22->Value();
//    mreal t_bh_pr_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_bh_pr_11.setZero();
//    tpe_bh_pr_11->Differential(DE_bh_pr_11);
//    mreal Dt_bh_pr_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_bh_pr_12.setZero();
//    tpe_bh_pr_12->Differential(DE_bh_pr_12);
//    mreal Dt_bh_pr_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_bh_pr_22.setZero();
//    tpe_bh_pr_22->Differential(DE_bh_pr_22);
//    mreal Dt_bh_pr_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    //######################################
//
//    tic();
//    E_fm_nl_11 = tpe_fm_nl_11->Value();
//    mreal t_fm_nl_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    E_fm_nl_12 = tpe_fm_nl_12->Value();
//    mreal t_fm_nl_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    E_fm_nl_22 = tpe_fm_nl_22->Value();
//    mreal t_fm_nl_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_fm_nl_11.setZero();
//    tpe_fm_nl_11->Differential(DE_fm_nl_11);
//    mreal Dt_fm_nl_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_fm_nl_12.setZero();
//    tpe_fm_nl_12->Differential(DE_fm_nl_12);
//    mreal Dt_fm_nl_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_fm_nl_22.setZero();
//    tpe_fm_nl_22->Differential(DE_fm_nl_22);
//    mreal Dt_fm_nl_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    //######################################
//
//    tic();
//    E_fm_11 = tpe_fm_11->Value();
//    mreal t_fm_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    E_fm_12 = tpe_fm_12->Value();
//    mreal t_fm_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    E_fm_22 = tpe_fm_22->Value();
//    mreal t_fm_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_fm_11.setZero();
//    tpe_fm_11->Differential(DE_fm_11);
//    mreal Dt_fm_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_fm_12.setZero();
//    tpe_fm_12->Differential(DE_fm_12);
//    mreal Dt_fm_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_fm_22.setZero();
//    tpe_fm_22->Differential(DE_fm_22);
//    mreal Dt_fm_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//
//    //######################################
//    tic();
//    E_fm_pr_11 = tpe_fm_pr_11->Value();
//    mreal t_fm_pr_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    E_fm_pr_12 = tpe_fm_pr_12->Value();
//    mreal t_fm_pr_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    E_fm_pr_22 = tpe_fm_pr_22->Value();
//    mreal t_fm_pr_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_fm_pr_11.setZero();
//    tpe_fm_pr_11->Differential(DE_fm_pr_11);
//    mreal Dt_fm_pr_11 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_fm_pr_12.setZero();
//    tpe_fm_pr_12->Differential(DE_fm_pr_12);
//    mreal Dt_fm_pr_12 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    tic();
//    DE_fm_pr_22.setZero();
//    tpe_fm_pr_22->Differential(DE_fm_pr_22);
//    mreal Dt_fm_pr_22 = toc();
//    std::cout << "done " << ++counter << " / " << count << std::endl;
//
//    //######################################
//
//    int w1 = 21;
//    int w  = 13;
////
////        std::string line = "--------------------------------------------------------------------------------------------------------------------------------------------------------------------";
//    std::string line = std::string( 3 * energy_count + w1 + w * energy_count, '-');
//    std::cout   << std::left;
//    std::cout   << std::setw(w1) << ""
//                << " | " << std::setw(w) << "exact"
//                << " | " << std::setw(w) << "pr"
//                << " | " << std::setw(w) << "BH"
//                << " | " << std::setw(w) << "BH_pr"
//                << " | " << std::setw(w) << "FMM"
//                << " | " << std::setw(w) << "FMM_pr"
//                << " | " << std::setw(w) << "FMM_nl"
//                << std::endl;
//
//    std::cout   << line << std::endl;
//
//    std::cout   << std::setw(w1) << "  E_11 "
//                << " | " << std::setw(w) << E_ex_11
//                << " | " << std::setw(w) << E_ex_pr_11
//                << " | " << std::setw(w) << E_bh_11
//                << " | " << std::setw(w) << E_bh_pr_11
//                << " | " << std::setw(w) << E_fm_11
//                << " | " << std::setw(w) << E_fm_pr_11
//                << " | " << std::setw(w) << E_fm_nl_11
//                << std::endl;
//
//    std::cout   << std::setw(w1) << "  E_12 "
//                << " | " << std::setw(w) << E_ex_12
//                << " | " << std::setw(w) << E_ex_pr_12
//                << " | " << std::setw(w) << E_bh_12
//                << " | " << std::setw(w) << E_bh_pr_12
//                << " | " << std::setw(w) << E_fm_12
//                << " | " << std::setw(w) << E_fm_pr_12
//                << " | " << std::setw(w) << E_fm_nl_12
//                << std::endl;
//
//    std::cout   << std::setw(w1) << "  E_22 "
//                << " | " << std::setw(w) << E_ex_22
//                << " | " << std::setw(w) << E_ex_pr_22
//                << " | " << std::setw(w) << E_bh_22
//                << " | " << std::setw(w) << E_bh_pr_22
//                << " | " << std::setw(w) << E_fm_22
//                << " | " << std::setw(w) << E_fm_pr_22
//                << " | " << std::setw(w) << E_fm_nl_22
//                << std::endl;
//
//
//    std::cout   << "\n";
//    std::cout   << std::setw(w1) << ""
//                << " | " << std::setw(w) << "exact"
//                << " | " << std::setw(w) << "pr"
//                << " | " << std::setw(w) << "BH"
//                << " | " << std::setw(w) << "BH_pr"
//                << " | " << std::setw(w) << "FMM"
//                << " | " << std::setw(w) << "FMM_pr"
//                << " | " << std::setw(w) << "FMM_nl"
//                << std::endl;
//
//    std::cout   << line << std::endl;
//
//    std::cout   << std::setw(w1) << "  E_11 error (%) "
//                << " | " << std::setw(w) << fabs(E_ex_11 / E_ex_11 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_ex_pr_11 / E_ex_11 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_bh_11 / E_ex_11 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_bh_pr_11 / E_ex_11 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_fm_11 / E_ex_11 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_fm_pr_11 / E_ex_11 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_fm_nl_11 / E_ex_11 - 1) * 100
//                << std::endl;
//
//    std::cout   << std::setw(w1) << "  E_12 error (%) "
//                << " | " << std::setw(w) << fabs(E_ex_12 / E_ex_12 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_ex_pr_12 / E_ex_12 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_bh_12 / E_ex_12 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_bh_pr_12 / E_ex_12 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_fm_12 / E_ex_12 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_fm_pr_12 / E_ex_12 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_fm_nl_12 / E_ex_12 - 1) * 100
//                << std::endl;
//
//    std::cout   << std::setw(w1) << "  E_22 error (%) "
//                << " | " << std::setw(w) << fabs(E_ex_22 / E_ex_22 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_ex_pr_22 / E_ex_22 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_bh_22 / E_ex_22 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_bh_pr_22 / E_ex_22 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_fm_22 / E_ex_22 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_fm_pr_22 / E_ex_22 - 1) * 100
//                << " | " << std::setw(w) << fabs(E_fm_nl_22 / E_ex_22 - 1) * 100
//                << std::endl;
//
//    std::cout   << std::setw(w1) << " DE_11 error (%) "
//                << " | " << std::setw(w) << (DE_ex_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
//                << " | " << std::setw(w) << (DE_ex_pr_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
//                << " | " << std::setw(w) << (DE_bh_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
//                << " | " << std::setw(w) << (DE_bh_pr_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
//                << " | " << std::setw(w) << (DE_fm_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
//                << " | " << std::setw(w) << (DE_fm_pr_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
//                << " | " << std::setw(w) << (DE_fm_nl_11 - DE_ex_11).norm() / DE_ex_11.norm() * 100
//                << std::endl;
//
//    std::cout   << std::setw(w1) << " DE_12 error (%) "
//                << " | " << std::setw(w) << (DE_ex_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
//                << " | " << std::setw(w) << (DE_ex_pr_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
//                << " | " << std::setw(w) << (DE_bh_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
//                << " | " << std::setw(w) << (DE_bh_pr_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
//                << " | " << std::setw(w) << (DE_fm_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
//                << " | " << std::setw(w) << (DE_fm_pr_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
//                << " | " << std::setw(w) << (DE_fm_nl_12 - DE_ex_12).norm() / DE_ex_12.norm() * 100
//                << std::endl;
//
//    std::cout   << std::setw(w1) << " DE_22 error (%) "
//                << " | " << std::setw(w) << (DE_ex_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
//                << " | " << std::setw(w) << (DE_ex_pr_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
//                << " | " << std::setw(w) << (DE_bh_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
//                << " | " << std::setw(w) << (DE_bh_pr_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
//                << " | " << std::setw(w) << (DE_fm_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
//                << " | " << std::setw(w) << (DE_fm_pr_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
//                << " | " << std::setw(w) << (DE_fm_nl_22 - DE_ex_22).norm() / DE_ex_22.norm() * 100
//                << std::endl;
//
//
//    std::cout   << "\n"
//                << std::endl;
//    std::cout   << std::setw(w1) << ""
//                << " | " << std::setw(w) << "exact"
//                << " | " << std::setw(w) << "pr"
//                << " | " << std::setw(w) << "BH"
//                << " | " << std::setw(w) << "BH_pr"
//                << " | " << std::setw(w) << "FMM"
//                << " | " << std::setw(w) << "FMM_pr"
//                << " | " << std::setw(w) << "FMM_nl"
//                << std::endl;
//
//    std::cout   << line << std::endl;
//
//    std::cout   << std::setw(w1) << "  E_11 time  (s) "
//                << " | " << std::setw(w) << t_ex_11
//                << " | " << std::setw(w) << t_ex_pr_11
//                << " | " << std::setw(w) << t_bh_11
//                << " | " << std::setw(w) << t_bh_pr_11
//                << " | " << std::setw(w) << t_fm_11
//                << " | " << std::setw(w) << t_fm_pr_11
//                << " | " << std::setw(w) << t_fm_nl_11
//                << std::endl;
//
//    std::cout   << std::setw(w1) << "  E_12 time  (s) "
//                << " | " << std::setw(w) << t_ex_12
//                << " | " << std::setw(w) << t_ex_pr_12
//                << " | " << std::setw(w) << t_bh_12
//                << " | " << std::setw(w) << t_bh_pr_12
//                << " | " << std::setw(w) << t_fm_12
//                << " | " << std::setw(w) << t_fm_pr_12
//                << " | " << std::setw(w) << t_fm_nl_12
//                << std::endl;
//
//    std::cout   << std::setw(w1) << "  E_22 time  (s) "
//                << " | " << std::setw(w) << t_ex_22
//                << " | " << std::setw(w) << t_ex_pr_22
//                << " | " << std::setw(w) << t_bh_22
//                << " | " << std::setw(w) << t_bh_pr_22
//                << " | " << std::setw(w) << t_fm_22
//                << " | " << std::setw(w) << t_fm_pr_22
//                << " | " << std::setw(w) << t_fm_nl_22
//                << std::endl;
//
//    std::cout   << std::setw(w1) << " DE_11 time  (s) "
//                << " | " << std::setw(w) << Dt_ex_11
//                << " | " << std::setw(w) << Dt_ex_pr_11
//                << " | " << std::setw(w) << Dt_bh_11
//                << " | " << std::setw(w) << Dt_bh_pr_11
//                << " | " << std::setw(w) << Dt_fm_11
//                << " | " << std::setw(w) << Dt_fm_pr_11
//                << " | " << std::setw(w) << Dt_fm_nl_11
//                << std::endl;
//
//    std::cout   << std::setw(w1) << " DE_12 time  (s) "
//                << " | " << std::setw(w) << Dt_ex_12
//                << " | " << std::setw(w) << Dt_ex_pr_12
//                << " | " << std::setw(w) << Dt_bh_12
//                << " | " << std::setw(w) << Dt_bh_pr_12
//                << " | " << std::setw(w) << Dt_fm_12
//                << " | " << std::setw(w) << Dt_fm_pr_12
//                << " | " << std::setw(w) << Dt_fm_nl_12
//                << std::endl;
//
//    std::cout   << std::setw(w1) << " DE_22 time  (s) "
//                << " | " << std::setw(w) << Dt_ex_22
//                << " | " << std::setw(w) << Dt_ex_pr_22
//                << " | " << std::setw(w) << Dt_bh_22
//                << " | " << std::setw(w) << Dt_bh_pr_22
//                << " | " << std::setw(w) << Dt_fm_22
//                << " | " << std::setw(w) << Dt_fm_pr_22
//                << " | " << std::setw(w) << Dt_fm_nl_22
//                << std::endl;



    return EXIT_SUCCESS;
}
