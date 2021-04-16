#include "main2.h"

using namespace rsurfaces;
using namespace geometrycentral;
using namespace geometrycentral::surface;

int main(int arg_count, char* arg_vec[])
{
    using namespace rsurfaces;
    
    namespace po = boost::program_options;
    
    auto BM = Benchmarker();
    
#pragma omp parallel
    {
        BM.thread_count = BM.max_thread_count = omp_get_num_threads();
    }
    
    po::options_description desc("Allowed options");
    
    try
    {
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
        ("thread_count", po::value<mint>(), "number of threads to be used")
        ("thread_step", po::value<mint>(), "increase number of threads by this in each iteration")
        
        ("burn_ins", po::value<mint>(), "number of burn-in iterations to use")
        ("iterations", po::value<mint>(), "number of iterations to use for the benchmark")
        
        ("tree_perc_alg", po::value<mint>(), "algorithm used for tree percolation. Possible values are 0 (sequential algorithm), 1 (using OpenMP tasks -- no scalable!), and 2 (an attempt to achieve better scalability).");
        
        po::variables_map var_map;
        po::store(po::parse_command_line(arg_count, arg_vec, desc), var_map);
        po::notify(var_map);
        
        if( var_map.count("help") )
        {
            std::cout << desc << "\n";
            return 0;
        }
        
        if( var_map.count("mesh") )
        {
            BM.obj1 = var_map["mesh"].as<std::string>();
        }
        if( var_map.count("profile_name") )
        {
            BM.profile_name = var_map["profile_name"].as<std::string>();
        }
        if( var_map.count("profile_path") )
        {
            BM.profile_path = var_map["profile_path"].as<std::string>();
        }
        
        if( var_map.count("theta") )
        {
            BM.theta = var_map["theta"].as<mreal>();
        }
        if( var_map.count("chi") )
        {
            BM.chi = var_map["chi"].as<mreal>();
        }
        if( var_map.count("alpha") )
        {
            BM.alpha = var_map["alpha"].as<mreal>();
        }
        if( var_map.count("beta") )
        {
            BM.beta = var_map["beta"].as<mreal>();
        }
        
        if( var_map.count("thread_count") )
        {
            BM.thread_count = var_map["thread_count"].as<mint>();
            BM.max_thread_count = var_map["thread_count"].as<mint>();
        }
        if( var_map.count("thread_step") )
        {
            BM.thread_step = var_map["thread_step"].as<mint>();
        }
        if( var_map.count("burn_ins"))
        {
            BM.burn_ins = var_map["burn_ins"].as<mint>();
        }
        if( var_map.count("iterations") )
        {
            BM.iterations = var_map["iterations"].as<mint>();
        }
        switch( var_map["tree_perc_alg"].as<mint>() ) {
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
    BM.PrepareVectors();
    
    for( mint threads = 0; threads < BM.max_thread_count + 1; threads += BM.thread_step )
    {
        // 0 is the new 1. (Want to have steps, but also  a single-threaded run.
        if( threads == 1)
        {
            break;
        }
        if( threads == 0)
        {
            BM.thread_count = 1;
        }
        else
        {
            BM.thread_count = threads;
        }
        
        std::cout << std::endl;
        std::cout << "### threads =  " << BM.thread_count << std::endl;

        omp_set_num_threads(BM.thread_count);
        mkl_set_num_threads(BM.thread_count);
        
        
        {
            mint a , b, c, d, e;
            #pragma omp parallel
            {
                a = omp_get_num_threads();
                c = mkl_get_max_threads();
            }
            b = omp_get_num_threads();
            d = mkl_get_max_threads();
//            e = tbb::task_scheduler_init::default_num_threads();
            std::cout << "omp_get_num_threads() in omp parallel = " << a << std::endl;
            std::cout << "omp_get_num_threads() in omp parallel = " << b << std::endl;

            std::cout << "mkl_get_max_threads() = " << c << std::endl;
            std::cout << "mkl_get_max_threads() = " << d << std::endl;
            
//            std::cout << "tbb::task_scheduler_init::default_num_threads() = " << e << std::endl;
        }

        ClearProfile(BM.profile_path + "/" + BM.profile_name + "_" + std::to_string(BM.thread_count) + ".tsv");
        
        //burn-in
        for( mint i = 0; i < BM.burn_ins; ++i)
        {
            std::cout << "burn_in " << i+1 << " / " << BM.burn_ins << std::endl;
            BM.Compute(-1);

        }
        
        ClearProfile(BM.profile_path + "/" + BM.profile_name + "_" + std::to_string(BM.thread_count) + ".tsv");
        
        //the actual test code
        for( mint i = 0; i < BM.iterations; ++i)
        {
            std::cout << "iterations " << i+1 << " / " << BM.iterations << std::endl;
            ptic("Iteration");
            BM.Compute(i);
            ptoc("Iteration");
            
        }
        std::cout << std::endl;
    }
    return EXIT_SUCCESS;
}
