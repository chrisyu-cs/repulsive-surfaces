#include "main_consistency.h"

using namespace rsurfaces;
using namespace geometrycentral;
using namespace geometrycentral::surface;

struct ConsistencyTest
{
    mreal alpha;
    mreal beta;
    std::vector<mreal> theta_list;
    std::shared_ptr<SurfaceEnergy> energy;
    
    MeshPtr mesh;
    GeomPtr geom;

    
    std::ofstream outfile;
    std::string title;
    
    mreal max_edge_length;
    mreal mean_edge_length;
    mreal median_edge_length;
    mint triangle_count;
    mreal theta;
    mreal energy_value;
    mreal disc_energy_value;
    mreal true_energy_value;
    mreal construction_time;
    mreal evaluation_time;
    mreal disc_evaluation_time;
    

    void print()
    {
        outfile
        << max_edge_length << "\t"
        << theta << "\t"
        << energy_value << "\t"
        << std::abs(1. - energy_value/disc_energy_value) << "\t"
        << std::abs(1. - energy_value/true_energy_value) << "\t"
        << construction_time << "\t"
        << evaluation_time  << "\t"
        << disc_evaluation_time/evaluation_time  << "\t"
        << mean_edge_length << "\t"
        << median_edge_length << "\t"
        << triangle_count
        << std::endl;
    }
    
    void process()
    {
            outfile << "***" << "\t" <<  title << "\t" << true_energy_value << std::endl;
            outfile
                << "Max edge length" << "\t"
                << "theta" << "\t"
                << "Value" << "\t"
                << "Relative error to discrete" << "\t"
                << "Relative error to true" << "\t"
                << "Construction time" << "\t"
                << "Evaluation time" << "\t"
                << "Speedup" << "\t"
                << "Mean edge length" << "\t"
                << "Median edge length" << "\t"
                << "Triangle count"
                << std::endl;
        
        tic("Creating all pairs energy");
        energy = std::make_shared<TPEnergyAllPairs>(mesh, geom, alpha, beta);
        toc("Creating all pairs energy");
        
        tic("Evaluating all pairs energy");
        disc_energy_value = energy->Value();
        disc_evaluation_time = toc("Evaluating pairs energy");
    
        
        for( size_t i = 0; i < theta_list.size();  ++i )
        {
            theta = theta_list[i];

            std::cout << "theta = " << theta << std::endl;
            
            tic("Creating " + title);
            energy = std::make_shared<TPEnergyBarnesHut0>(mesh, geom, alpha, beta, theta);
            construction_time = toc("Evaluating " + title);
            
            tic("Evaluating " + title);
            energy_value = energy->Value();
            evaluation_time = toc("Evaluating "  + title);
            
            print();
        }
        
        construction_time = 0.;
        theta = 0.;
        evaluation_time = disc_evaluation_time;
        energy_value = disc_energy_value;
        
        print();
        
    }
}; // ConsistencyTest


int main(int arg_count, char* arg_vec[])
{
    using namespace rsurfaces;
    
    std::string input_filename;
    std::string output_filename;
    
    ConsistencyTest test;
    
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
        boost::filesystem::path q { p.parent_path() / (p.stem().string() + "_Results.tsv") };
        output_filename = q.string();
    }
    
    
    test.alpha = 6.;
    test.beta = 12.;
    test.theta_list = std::vector<mreal>  { 2., sqrt(2.), 1., 1./sqrt(2.), 0.5, 0.5/sqrt(2.), 0.25, 0.25 /sqrt(2.), 0.125 };
    
    MeshUPtr umesh;
    GeomUPtr ungeom0;
    GeomUPtr ungeom1;
    GeomUPtr ungeom2;
    
    std::cout << "Reading file " + input_filename + "..." << std::endl;
    std::tie(umesh, ungeom0, ungeom1 /*, ungeom2*/) = readMeshWithNormals(input_filename);
    test.mesh = std::move(umesh);
    
    std::string meta_filename;
    {
        boost::filesystem::path p (input_filename);
        boost::filesystem::path q { p.parent_path() / (p.stem().string() + "_Meta.tsv") };
        meta_filename = q.string();
    }
    std::cout << "Reading file " + meta_filename + "..." << std::endl;
    {
        std::ifstream in(meta_filename);
        mreal value;
        std::string line;
        std::string token;
        while (getline(in, line))
        {
            std::stringstream ss(line);
            ss >> token;
            if (token == "\"TrueEnergyValue\"") {
                ss >> test.true_energy_value;
//                std::cout << token << " = " << test.true_energy_value << std::endl;
            } else if (token == "\"MaxEdgeLength\"") {
                ss >> test.max_edge_length;
//                std::cout << token << " = " << test.max_edge_length << std::endl;
            } else if (token == "\"MeanEdgeLength\"") {
                ss >> test.mean_edge_length;
//                std::cout << token << " = " << test.mean_edge_length << std::endl;
            } else if (token == "\"MedianEdgeLength\"") {
                ss >> test.median_edge_length;
//                std::cout << token << " = " << test.median_edge_length << std::endl;
            } else if (token == "\"TriangleCount\"") {
                ss >> test.triangle_count;
//                std::cout << token << " = " << test.triangle_count << std::endl;
            }
        }
        in.close();
    }

    omp_set_num_threads(1);
    
    test.outfile.open(output_filename);
    test.outfile << std::setprecision(16);
    
    test.geom = std::move(ungeom0);
    test.title = "Barnes-Hut energy";
    test.process();
    
    test.geom = std::move(ungeom1);
    test.title = "Barnes-Hut energy with 2nd order face normals";
    test.process();
    
//    test.geom = std::move(ungeom2);
//    test.title = "Barnes-Hut energy with exact vertex normals";
//    test.process();
    
    
    test.outfile.close();
    

    
//    output_filename = "/Users/Henrik/a.tsv";
//    std::cout << output_filename << std::endl;
//    outfile.open(output_filename);
//
//    for( auto v : mesh->vertices() )
//    {
//        Vector3 vec = geom->inputVertexPositions[v.getIndex()];
//        Vector3 nor = geom->vertexNormals[v.getIndex()];
//        outfile << vec.x << "\t" << vec.y << "\t" << vec.z << "\t" << nor.x << "\t" << nor.y << "\t" << nor.z << std::endl;
//    }
//    outfile.close();
//
//    output_filename = "/Users/Henrik/b.tsv";
//    outfile.open(output_filename);
//    for( auto v : mesh->vertices() )
//    {
//        Vector3 vec = ngeom->inputVertexPositions[v.getIndex()];
//        Vector3 nor = ngeom->vertexNormals[v.getIndex()];
//        outfile << vec.x << "\t" << vec.y << "\t" << vec.z << "\t" << nor.x << "\t" << nor.y << "\t" << nor.z << std::endl;
//    }
//    outfile.close();
//
//    output_filename = "/Users/Henrik/c.tsv";
//    outfile.open(output_filename);
//
//    FaceIndices fInds = mesh->getFaceIndices();
//    VertexIndices vInds = mesh->getVertexIndices();
//
//
//    for (auto face : mesh->faces())
//    {
//        int i = fInds[face];
//
//        GCHalfedge he = face.halfedge();
//
//        int i0 = vInds[he.vertex()];
//        int i1 = vInds[he.next().vertex()];
//        int i2 = vInds[he.next().next().vertex()];
//
//        outfile << i0 << "\t" << i1 << "\t" << i2<< std::endl;
//    }
//    outfile.close();
//
//    output_filename = "/Users/Henrik/d.tsv";
//    outfile.open(output_filename);
//
//
//    for (auto face : mesh->faces())
//    {
//        int i = fInds[face];
//
//        GCHalfedge he = face.halfedge();
//
//        int i0 = vInds[he.vertex()];
//        int i1 = vInds[he.next().vertex()];
//        int i2 = vInds[he.next().next().vertex()];
//
//        Vector3 vec1 = geom->faceNormals[face.getIndex()];
//        Vector3 vec2 = ngeom->faceNormals[face.getIndex()];
//        if( geom->faceAreas[face.getIndex()]  != ngeom->faceAreas[face.getIndex()] )
//        {
//            print("!!!");
//        }
//
//        outfile << vec1.x << "\t" << vec1.y << "\t" << vec1.z << "\t" << vec2.x << "\t" << vec2.y << "\t" << vec2.z << std::endl;
//    }
//    outfile.close();
    
    std::cout << "Done." << std::endl;
    
    return EXIT_SUCCESS;
}
