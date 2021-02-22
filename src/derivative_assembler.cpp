
#include "derivative_assembler.h"

namespace rsurfaces
{
    Eigen::SparseMatrix<double> DerivativeAssembler(MeshPtr const & mesh, GeomPtr const & geom)
    {
        FaceIndices fInds = mesh->getFaceIndices();
        VertexIndices vInds = mesh->getVertexIndices();

        int vertex_count = mesh->nVertices();
        int tuple_count = mesh->nFaces();
        int tuple_length = 3;
        int emb_dim = 3;

        auto outer  = std::vector<int> ( tuple_count * tuple_length + 1 , 0);
        auto inner  = std::vector<int> ( tuple_count * tuple_length, 0);
        auto values = std::vector<double>( tuple_count * tuple_length, 0.);

        for( auto face : mesh->faces() )
        {
            int i = fInds[face];

            GCHalfedge he = face.halfedge();

            int i0 = vInds[he.vertex()];
            int i1 = vInds[he.next().vertex()];
            int i2 = vInds[he.next().next().vertex()];
            
            outer[tuple_length * i + 1] = tuple_length * i + 1;
            outer[tuple_length * i + 2] = tuple_length * i + 2;
            outer[tuple_length * i + 3] = tuple_length * i + 3;
            
            inner[tuple_length * i + 0] = i0;
            inner[tuple_length * i + 1] = i1;
            inner[tuple_length * i + 2] = i2;
            
            values[tuple_length * i + 0] = 1.;
            values[tuple_length * i + 1] = 1.;
            values[tuple_length * i + 2] = 1.;
        }
        return Eigen::Map<Eigen::SparseMatrix<double>> ( vertex_count, tuple_count * tuple_length, tuple_count * tuple_length, &outer[0], &inner[0], &values[0] );
    }
} // namespace rsurfaces
