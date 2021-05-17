#include "main_runtest.h"

using namespace rsurfaces;
using namespace geometrycentral;
using namespace geometrycentral::surface;

int main(int arg_count, char* arg_vec[])
{
    using namespace rsurfaces;
    
    auto BM = Benchmarker();
    
#pragma omp parallel
    {
        BM.thread_count = BM.max_thread_count = omp_get_num_threads();
    }
    
#ifdef PROFILING
    std::cout << "Profiling activated." << std::endl;
#else
    std::cout << "Profiling deactivated." << std::endl;
#endif
    
#ifdef MKL_DIRECT_CALL_SEQ_JIT
    std::cout << "Called with MKL_DIRECT_CALL_SEQ_JIT activated." << std::endl;
#else
    std::cout << "Called with MKL_DIRECT_CALL_SEQ_JIT deactivated." << std::endl;
#endif
    
    std::cout << "Using Eigen version " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;
    
    MKLVersion Version;
    mkl_get_version(&Version);
    
    std::cout << "Using MKL version " << Version.MajorVersion << "." << Version.MinorVersion << "." << Version.UpdateVersion << std::endl;
    
    
    args::ArgumentParser parser("geometry-central & Polyscope example project");

    args::ValueFlag<std::string> mesh_Flag(parser, "mesh", "file of mesh to use as variable", {"mesh"});
    args::ValueFlag<std::string> path_Flag(parser, "path", "path to store data", {"path"});
    args::ValueFlag<std::string> profile_name_Flag(parser, "profile_name", "file base name of profile file", {"profile_name"});
    
    args::ValueFlag<mreal> alpha_Flag(parser, "alpha", "first TP parameter (numerator)", {"alpha"});
    args::ValueFlag<mreal> beta_Flag(parser, "beta", "second TP parameter (denominator)", {"beta"});
    args::ValueFlag<mreal> theta_Flag(parser, "theta", "separation parameter for barnes-hut method", {"theta"});
    args::ValueFlag<mreal> chi_Flag(parser, "chi", "separation parameter for block cluster tree", {"chi"});
    args::ValueFlag<mint> thread_Flag(parser, "threads", "number of threads to be used", {"threads"});
    args::ValueFlag<mint> split_threshold_Flag(parser, "split_threshold", "maximal number of primitives per leaf cluster", {"split_threshold"});
    
    args::ValueFlag<mint> thread_step_Flag(parser, "thread_step", "increase number of threads by this in each iteration", {"thread_step"});
    args::ValueFlag<mint> burn_ins_Flag(parser, "burn_ins", "number of burn-in iterations to use", {"burn_ins"});
    args::ValueFlag<mint> iterations_Flag(parser, "iterations", "number of iterations to use for the benchmark", {"iterations"});
    args::ValueFlag<mint> tree_perc_alg_Flag(parser, "tree_perc_alg", "algorithm used for tree percolation. Possible values are 0 (sequential algorithm), 1 (using OpenMP tasks -- no scalable!), and 2 (an attempt to achieve better scalability)", {"tree_perc_alg"});
    
    // Parse args
    try
    {
        parser.ParseCLI(arg_count, arg_vec);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    if (mesh_Flag)
    {
        BM.obj1 = args::get(mesh_Flag);
    }
    if (profile_name_Flag)
    {
        BM.profile_name = args::get(profile_name_Flag);
    }
    if (path_Flag)
    {
        BM.path = args::get(path_Flag);
    }

    

    if (theta_Flag)
    {
        BM.theta = args::get(theta_Flag);
    }
    if (chi_Flag)
    {
        BM.chi = args::get(chi_Flag);
    }
    

    if (alpha_Flag)
    {
        BM.alpha = args::get(alpha_Flag);
    }
    if (beta_Flag)
    {
        BM.beta = args::get(beta_Flag);
    }

    if (thread_Flag)
    {
        BM.thread_count = args::get(thread_Flag);
        BM.max_thread_count = args::get(thread_Flag);
    }
    
    
    if (split_threshold_Flag)
    {
        BVHDefaultSettings.split_threshold = args::get(split_threshold_Flag);
    }
    if (thread_step_Flag)
    {
        BM.thread_step = args::get(thread_step_Flag);
    }
    if (burn_ins_Flag)
    {
        BM.burn_ins = args::get(burn_ins_Flag);
    }
    if (iterations_Flag)
    {
        BM.iterations = args::get(iterations_Flag);
    }
    if (tree_perc_alg_Flag)
    {
        switch( args::get(tree_perc_alg_Flag) )
        {
            case 1:
                BM.tree_perc_alg = TreePercolationAlgorithm::Tasks;
                break;
            case 2:
                BM.tree_perc_alg = TreePercolationAlgorithm::Chunks;
                break;
            case 0:
                BM.tree_perc_alg = TreePercolationAlgorithm::Sequential;
                break;
            default:
                BM.tree_perc_alg = TreePercolationAlgorithm::Tasks;
                break;
        }
    }
        
    
    std::cout << std::setprecision(8);
    
    MeshUPtr u_mesh;
    GeomUPtr u_geom;
    
    // Load mesh1
    std::cout << "Loading " << BM.obj1 << " as variable." << std::endl;
    std::tie(u_mesh, u_geom) = readMesh(BM.obj1);
    BM.mesh1 = std::move(u_mesh);
    BM.geom1 = std::move(u_geom);
    
    BM.thread_count = BM.max_thread_count;
    omp_set_num_threads(BM.thread_count);
    mkl_set_num_threads(BM.thread_count);
    BM.PrintStats();
    
    BM.Prepare();

    BM.TestDerivatives();
    
    
    
//    BM.TestMultiply();
//    BM.TestMKLOptimize();
//    BM.TestVBSR();
//    BM.TestHybrid();

//    BM.TestBatch();

//    BM.TestPrePost();
    
//    for( mint threads = 0; threads < BM.max_thread_count + 1; threads += BM.thread_step )
//    {
//        // 0 is the new 1. (Want to have steps, but also  a single-threaded run.
//        if( threads == 1)
//        {
//            break;
//        }
//        if( threads == 0)
//        {
//            BM.thread_count = 1;
//        }
//        else
//        {
//            BM.thread_count = threads;
//        }
//
//        std::cout << std::endl;
//        std::cout << "### threads =  " << BM.thread_count << std::endl;
//
//        omp_set_num_threads(BM.thread_count);
//        mkl_set_num_threads(BM.thread_count);
//
//
//        {
//            mint a , b, c, d, e;
//            #pragma omp parallel
//            {
//                a = omp_get_num_threads();
//                c = mkl_get_max_threads();
//            }
//            b = omp_get_num_threads();
//            d = mkl_get_max_threads();
////            e = tbb::task_scheduler_init::default_num_threads();
//            std::cout << "omp_get_num_threads() in omp parallel = " << a << std::endl;
//            std::cout << "omp_get_num_threads() in omp parallel = " << b << std::endl;
//
//            std::cout << "mkl_get_max_threads() = " << c << std::endl;
//            std::cout << "mkl_get_max_threads() = " << d << std::endl;
//
////            std::cout << "tbb::task_scheduler_init::default_num_threads() = " << e << std::endl;
//        }
//
//        ClearProfile(BM.path + "/" + BM.profile_name + "_" + std::to_string(BM.thread_count) + ".tsv");
//
//
//
//        //burn-in
//        for( mint i = 0; i < BM.burn_ins; ++i)
//        {
//            std::cout << "burn_in " << i+1 << " / " << BM.burn_ins << std::endl;
//            BM.Compute(-1);
//
//        }
//
//        ClearProfile(BM.path + "/" + BM.profile_name + "_" + std::to_string(BM.thread_count) + ".tsv");
//
//        //the actual test code
//        for( mint i = 0; i < BM.iterations; ++i)
//        {
//            std::cout << "iterations " << i+1 << " / " << BM.iterations << std::endl;
//            ptic("Iteration");
//            BM.Compute(i);
//            ptoc("Iteration");
//
//        }
//        std::cout << std::endl;
//    }
    
    return EXIT_SUCCESS;
}
