#include "consistency_helpers.h"

using namespace rsurfaces;
using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace  rsurfaces
{
    

    
//    namespace {
//
//
//        // String manipulation helpers to parse .obj files
//        // See http://stackoverflow.com/a/236803
//        // ONEDAY: move to utility?
//        std::vector<std::string>& split(const std::string& s, char delim, std::vector<std::string>& elems) {
//            std::stringstream ss(s);
//            std::string item;
//            while (std::getline(ss, item, delim)) {
//                elems.push_back(item);
//            }
//            return elems;
//        }
//        std::vector<std::string> split(const std::string& s, char delim) {
//            std::vector<std::string> elems;
//            split(s, delim, elems);
//            return elems;
//        }
//
//        class Index {
//        public:
//            Index() {}
//
//            Index(long long int v, long long int vt, long long int vn) : position(v), uv(vt), normal(vn) {}
//
//            bool operator<(const Index& i) const {
//                if (position < i.position) return true;
//                if (position > i.position) return false;
//                if (uv < i.uv) return true;
//                if (uv > i.uv) return false;
//                if (normal < i.normal) return true;
//                if (normal > i.normal) return false;
//
//                return false;
//            }
//
//            long long int position = -1;
//            long long int uv = -1;
//            long long int normal = -1;
//        };
//
//        Index parseFaceIndex(const std::string& token)
//        {
//            std::stringstream in(token);
//            std::string indexString;
//            int indices[3] = {1, 1, 1};
//
//            int i = 0;
//            while (std::getline(in, indexString, '/')) {
//                if (indexString != "\\") {
//                    std::stringstream ss(indexString);
//                    ss >> indices[i++];
//                }
//            }
//
//            // decrement since indices in OBJ files are 1-based
//            return Index(indices[0] - 1, indices[1] - 1, indices[2] - 1);
//        }
//
//        std::vector<std::string> supportedMeshTypes = {"obj", "ply", "stl", "off"};
//    }
    
//    std::tuple<MeshUPtr, GeomUPtr, GeomUPtr> readMeshWithNormals(std::string filename)
//    {
//        tic("readMeshWithNormals");
//        std::vector<Vector3> vertexCoordinates;
//        std::vector<Vector3> vertexNormals;
//        std::vector<std::vector<size_t>> polygons;
//
//        std::ifstream in(filename);
//        print("A");
//        // parse obj format
//        Vector3 vec;
//        std::string line;
//        std::string token;
//        while (getline(in, line))
//        {
//            std::stringstream ss(line);
//            ss >> token;
//            if (token == "v") {
//                ss >> vec;
//                vertexCoordinates.push_back(vec);
//            } else if (token == "vt") {
//                // do nothing
//            } else if (token == "vn") {
//                ss >> vec;
//                vertexNormals.push_back(vec);
//            } else if (token == "f") {
//                std::vector<size_t> face;
//                std::vector<size_t> faceCoordInds;
//                while (ss >> token) {
//                    Index index = parseFaceIndex(token);
//                    if (index.position < 0) {
//                        getline(in, line);
//                        size_t i = line.find_first_not_of("\t\n\v\f\r ");
//                        index = parseFaceIndex(line.substr(i));
//                    }
//                    face.push_back(index.position);
//                }
//                polygons.push_back(face);
//            }
//        }
//        print("B");
//        in.close();
//
//        MeshUPtr mesh;
//        mesh.reset(new ManifoldSurfaceMesh(polygons));
//
//        GeomUPtr geom(new VertexPositionGeometry(*mesh));
//        print("C");
//        for ( Vertex v : mesh->vertices() ) {
//            // Use the low-level indexers here since we're constructing
//            geom->inputVertexPositions[v] = vertexCoordinates[v.getIndex()];
//        }
//
//        geom->requireFaceAreas();
//        geom->requireFaceNormals();
//        geom->requireVertexNormals();
//
//        GeomUPtr ngeom = geom->copy();
//
//        ngeom->requireFaceAreas();
//        ngeom->requireFaceNormals();
//        ngeom->requireVertexNormals();
//
//        print("D");
//        for ( Vertex v : mesh->vertices() ) {
//            // Use the low-level indexers here since we're constructing
//            ngeom->vertexNormals[v] = vertexNormals[v.getIndex()];
//        }
//
//        FaceIndices fInds = mesh->getFaceIndices();
//        VertexIndices vInds = mesh->getVertexIndices();
//        print("E");
////        for ( Face f : mesh->faces())
////        {
////            int i = fInds[f];
////
////            GCHalfedge he = f.halfedge();
////
//////            int i0 = vInds[he.vertex()];
//////            int i1 = vInds[he.next().vertex()];
//////            int i2 = vInds[he.next().next().vertex()];
////            Vector3 n1 = ngeom->vertexNormals[he.vertex()];
////            Vector3 n2 = ngeom->vertexNormals[he.next().vertex()];
////            Vector3 n3 = ngeom->vertexNormals[he.next().next().vertex()];
//////            Vector3 n = ( n1 + n2 + n3 );
//////            ngeom->faceNormals[f] = n/n.norm();
//////            ngeom->faceNormals[i] = ( n1 + n2 + n3 )/3.;
////            ngeom->faceNormals[f] = ( n1 + n2 + n3 )/3.;
////        }
//
//        boost::filesystem::path p (input_filename);
//        boost::filesystem::path q { p.parent_path() / (p.stem().string() + "_FaceNormals.tsv") };
//        in.open(q.string());
//        print("F");
//        // parse tsv format
//        std::vector<Vector3> faceNormals;
//        while (getline(in, line))
//        {
//            std::stringstream ss(line);
//            ss >> token;
//            ss >> vec;
//            faceNormals.push_back(vec);        }
//
//        for ( Face f : mesh->faces())
//        {
//            ngeom->faceNormals[f] = faceNormals[f.getIndex()];
//        }
//
//        toc("readMeshWithNormals");
//        return std::make_tuple(std::move(mesh), std::move(geom), std::move(ngeom));
//    }
    
} // namespace  rsurfaces
