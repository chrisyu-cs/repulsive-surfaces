#include "sobolev/constraints/vertex_normal.h"
#include "helpers.h"

namespace rsurfaces
{
    namespace Constraints
    {
        VertexNormalConstraint::VertexNormalConstraint(MeshPtr &mesh, GeomPtr &geom) {}

        void VertexNormalConstraint::pinVertices(MeshPtr &mesh, GeomPtr &geom, std::vector<size_t> &indices_)
        {
            for (size_t i = 0; i < indices_.size(); i++)
            {
                indices.push_back(indices_[i]);
                initNormals.push_back(vertexAreaNormalUnnormalized(geom, mesh->vertex(indices_[i])));
            }
        }

        void VertexNormalConstraint::addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            // TODO
        }

        void VertexNormalConstraint::addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            // TODO
        }

        size_t VertexNormalConstraint::nRows()
        {
            // Every vertex gets 3 rows, one to freeze each of its coordinates
            return indices.size() * 3;
        }

        void VertexNormalConstraint::ProjectConstraint(MeshPtr &mesh, GeomPtr &geom)
        {
            return;
        }

    } // namespace Constraints
} // namespace rsurfaces