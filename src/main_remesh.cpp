#include "main_remesh.h"

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
        boost::filesystem::path q { p.parent_path() / (p.stem().string() + "_Remeshed.obj") };
        output_filename = q.string();
    }
        
    MeshUPtr umesh;
    GeomUPtr ugeom;
    
    std::cout << "Reading file " + input_filename + "..." << std::endl;
    std::tie(umesh, ugeom) = readMesh(input_filename);
    MeshPtr mesh = std::move(umesh);
    GeomPtr geom = std::move(ugeom);
    GeomPtr ogeom = geom->copy();
    std::cout << "Remeshing..." << std::endl;
    
    auto remesher = rsurfaces::remeshing::DynamicRemesher( mesh, geom, ogeom );
    
    remesher.remeshingMode = remeshing::RemeshingMode::FlipOnly;
    remesher.smoothingMode = remeshing::SmoothingMode::Laplacian;
    remesher.flippingMode = remeshing::FlippingMode::Delaunay;
    
    remesher.Remesh(10, true );
    
    
    std::cout << "Writing file "+ output_filename + "..." << std::endl;
    writeMesh(mesh, geom, output_filename);
    std::cout << "Done." << std::endl;
    return EXIT_SUCCESS;
}
