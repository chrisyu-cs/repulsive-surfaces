#include "main_remesh.h"
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <regex>

using namespace rsurfaces;
using namespace geometrycentral;
using namespace geometrycentral::surface;


int main(int arg_count, char* arg_vec[])
{
    using namespace rsurfaces;
    
    std::string input_filename;
    std::string output_filename;
    
    std::cout << "Using Eigen version " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;
    
    MKLVersion Version;
    mkl_get_version(&Version);
    
    std::cout << "Using MKL version " << Version.MajorVersion << "." << Version.MinorVersion << "." << Version.UpdateVersion << std::endl;
    
    
    args::ArgumentParser parser("geometry-central & Polyscope example project");

    args::Positional<std::string> input_Flag (parser, "mesh", "A mesh file.");
    args::ValueFlag<std::string> output_Flag(parser, "output", "name of output file", {"output"});
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
    
    // Make sure a mesh name was given
    if( input_Flag )
    {
        input_filename = args::get(input_Flag);
    }
    else
    {
        std::cerr << "Please specify a mesh file as argument" << std::endl;
        return EXIT_FAILURE;
    }
    
    
    if( output_Flag )
    {
        output_filename = args::get(output_Flag);
    }
    else
    {
        boost::filesystem::path p (input_filename);
        boost::filesystem::path q { p.parent_path() / (p.stem().string() + ".tsv") };
        output_filename = q.string();
    }
        
    
    
    mreal alpha = 6.;
    mreal beta = 12.;
    mreal weight = 1.;
    std::vector<mreal> theta_list { 1., 0.5, 0.25, 0.125 };
    
    MeshUPtr umesh;
    GeomUPtr ugeom;
    GeomUPtr ungeom;
    std::cout << "Reading file " + input_filename + "..." << std::endl;
    std::tie(umesh, ugeom, ungeom) = readMeshWithNormals(input_filename);
    MeshPtr mesh = std::move(umesh);
    GeomPtr geom = std::move(ugeom);
    GeomPtr ngeom = std::move(ungeom);
    GeomPtr ogeom = geom->copy();
    
//    FaceIndices fInds = mesh->getFaceIndices();
//
//    geom->requireFaceNormals();
//
////    for (auto face : mesh->faces())
////    {
////        int i = fInds[face];
////        geom->faceNormals[i].x = 1.;
////        geom->faceNormals[i].y = 0.;
////        geom->faceNormals[i].z = 0.;
////    }
//
//    geom->requireFaceNormals();
//
//    for (auto face : mesh->faces())
//    {
//        int i = fInds[face];
//        std::cout << geom->faceNormals[i].x  << " ";
//    }
//    std::cout << std::endl;
    
    std::ofstream outfile;
    outfile.open(output_filename);
    
    mreal energy_value;
    mreal construction_time;
    mreal evaluation_time;
    
    mreal true_energy_value;
    mreal true_construction_time;
    mreal true_evaluation_time;
    
    std::shared_ptr<SurfaceEnergy> energy;

    
    outfile << std::setprecision(16);
    
    outfile << "All-pairs energy" << std::endl;
    BVHDefaultSettings.split_threshold = 16;
    omp_set_num_threads(4);
    
    
    tic("Creating all pairs energy");
        energy = std::make_shared<TPEnergyAllPairs>(mesh, geom, alpha, beta);
    true_construction_time = toc("Creating all pairs energy");
    
    tic("Evaluating all pairs energy");
        true_energy_value = energy->Value();
    true_evaluation_time = toc("Evaluating pairs energy");
        
    
    outfile << std::endl;
    
    outfile << "Barnes-Hut energy" << std::endl;
    outfile << "theta" << "\t" << "Value" << "\t" << "Construction time" << "\t" << "Evaluation time" << std::endl;
    BVHDefaultSettings.split_threshold = 8;
    omp_set_num_threads(1);
    for( size_t i = 0; i < theta_list.size();  ++i )
    {
        mreal theta = theta_list[i];

        std::cout << "theta = " << theta << std::endl;
        
        tic("Creating all Barnes-Hut energy");
            energy = std::make_shared<TPEnergyBarnesHut0>(mesh, geom, alpha, beta, theta);
        construction_time = toc("Evaluating pairs energy");
        
        tic("Evaluating all Barnes-Hut energy");
            energy_value = energy->Value();
        evaluation_time = toc("Evaluating all Barnes-Hut energy");
        
        outfile << theta << "\t" << energy_value << "\t" << construction_time << "\t" << evaluation_time << std::endl;
    }
    outfile << 0. << "\t" << true_energy_value << "\t" << true_construction_time << "\t" << true_evaluation_time << std::endl;
    
    
    
    
    
    tic("Creating all pairs energy with better normals");
        energy = std::make_shared<TPEnergyAllPairs>(mesh, ngeom, alpha, beta);
    true_construction_time = toc("Creating all pairs energy with better normals");
    
    tic("Evaluating all pairs energy with better normals");
        true_energy_value = energy->Value();
    true_evaluation_time = toc("Evaluating pairs energy with better normals");
        
    
    outfile << std::endl;
    
    outfile << "Barnes-Hut energy with better normals" << std::endl;
    outfile << "theta" << "\t" << "Value" << "\t" << "Construction time" << "\t" << "Evaluation time" << std::endl;
    BVHDefaultSettings.split_threshold = 8;
    omp_set_num_threads(1);
    for( size_t i = 0; i < theta_list.size();  ++i )
    {
        mreal theta = theta_list[i];

        std::cout << "theta = " << theta << std::endl;
        
        tic("Creating all Barnes-Hut energy with better normals");
            energy = std::make_shared<TPEnergyBarnesHut0>(mesh, ngeom, alpha, beta, theta);
        construction_time = toc("Evaluating pairs energy with better normals");
        
        tic("Evaluating all Barnes-Hut energy with better normals");
            energy_value = energy->Value();
        evaluation_time = toc("Evaluating all Barnes-Hut energy with better normals");
        
        outfile << theta << "\t" << energy_value << "\t" << construction_time << "\t" << evaluation_time << std::endl;
    }
    outfile << 0. << "\t" << true_energy_value << "\t" << true_construction_time << "\t" << true_evaluation_time << std::endl;
    
    
    
    
    outfile << std::endl;
    
    
    outfile.close();
    return EXIT_SUCCESS;
}
