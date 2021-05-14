#pragma once

#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "../deps/polyscope/deps/args/args/args.hxx"

#include <omp.h>
#include <mkl.h>
#include <mkl_spblas.h>

//#include <tbb/task_scheduler_init.h>
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

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <regex>


namespace rsurfaces
{
    void writeMesh(MeshPtr mesh, GeomPtr geom, std::string output)
    {
        using namespace std;
        ofstream outfile;
        outfile.open(output);
        outfile << std::setprecision(16);
        VertexIndices inds = mesh->getVertexIndices();
        
        // Write all vertices in order
        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            GCVertex vert = mesh->vertex(i);
            Vector3 pos = geom->inputVertexPositions[vert];
            outfile << "v " << pos.x << " " << pos.y << " " << pos.z << endl;
        }
        // Write all face indices
        for (GCFace face : mesh->faces())
        {
            outfile << "f ";
            for (GCVertex adjVert : face.adjacentVertices())
            {
                // OBJ is 1-indexed
                int vertInd = inds[adjVert] + 1;
                outfile << vertInd;
                outfile << " ";
            }
            outfile << endl;
        }
        
        outfile << endl;
        outfile.close();
    } // writeMesh
    
    std::tuple<MeshUPtr, GeomUPtr, GeomUPtr> readMeshWithNormals(std::string filename)
    {
        tic("readMeshWithNormals");
        
        MeshUPtr mesh;
        GeomUPtr geom;
        GeomUPtr geom1;
        GeomUPtr geom2;
        
        std::tie(mesh, geom) = readMesh(filename);
 
        geom->requireFaceAreas();
        geom->requireFaceNormals();
        geom->requireVertexNormals();
        
        Vector3 vec;
        std::string line;
        std::string token;
        std::vector<Vector3> faceNormals;
        boost::filesystem::path p;
        boost::filesystem::path q;
        std::ifstream in;
        
        
        geom1 = geom->copy();
        geom1->requireFaceAreas();
        geom1->requireFaceNormals();
        geom1->requireVertexNormals();
    
        faceNormals.clear();
        p = boost::filesystem::path(filename);
        q = boost::filesystem::path { p.parent_path() / (p.stem().string() + "_FaceNormals.tsv") };
        in.open(q.string());
        while (getline(in, line))
        {
            std::stringstream ss(line);
//            ss >> token;
            ss >> vec;
            faceNormals.push_back(vec);
        }
        in.close();
        for ( auto f : mesh->faces())
        {
            geom1->faceNormals[f] = faceNormals[f.getIndex()];
        }
        
        
        
//        geom2 = geom->copy();
//        geom2->requireFaceAreas();
//        geom2->requireFaceNormals();
//        geom2->requireVertexNormals();
//
//        faceNormals.clear();
//        p = boost::filesystem::path(filename);
//        q = boost::filesystem::path { p.parent_path() / (p.stem().string() + "_FaceNormals2.tsv") };
//        in.open(q.string());
//        while (getline(in, line))
//        {
//            std::stringstream ss(line);
////            ss >> token;
//            ss >> vec;
//            faceNormals.push_back(vec);
//        }
//        in.close();
//        for ( auto f : mesh->faces())
//        {
//            geom2->faceNormals[f] = faceNormals[f.getIndex()];
//        }
        
        
        toc("readMeshWithNormals");
        return std::make_tuple(std::move(mesh), std::move(geom), std::move(geom1));
    }
    
} // namespace rsurfaces
