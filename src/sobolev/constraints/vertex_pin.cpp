#include "sobolev/constraints/vertex_pin.h"

namespace rsurfaces
{
    namespace Constraints
    {
        VertexPinConstraint::VertexPinConstraint(MeshPtr &mesh, GeomPtr &geom) {}

        void VertexPinConstraint::pinVertices(MeshPtr &mesh, GeomPtr &geom, std::vector<size_t> &indices_)
        {
            for (size_t i = 0; i < indices_.size(); i++)
            {
                indices.push_back(indices_[i]);
                initPositions.push_back(geom->inputVertexPositions[mesh->vertex(indices_[i])]);
            }
        }

        void VertexPinConstraint::addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            // All we do is put a 1 in the index for all pinned vertices
            for (size_t i = 0; i < indices.size(); i++)
            {
                triplets.push_back(Triplet(baseRow + 3 * i, 3 * indices[i], 1));
                triplets.push_back(Triplet(baseRow + 3 * i + 1, 3 * indices[i] + 1, 1));
                triplets.push_back(Triplet(baseRow + 3 * i + 2, 3 * indices[i] + 2, 1));
            }
        }

        void VertexPinConstraint::addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            // All we do is put a 1 in the index for all pinned vertices
            for (size_t i = 0; i < indices.size(); i++)
            {
                M(baseRow + 3 * i, 3 * indices[i]) = 1;
                M(baseRow + 3 * i + 1, 3 * indices[i] + 1) = 1;
                M(baseRow + 3 * i + 2, 3 * indices[i] + 2) = 1;
            }
        }

        size_t VertexPinConstraint::nRows()
        {
            // Every vertex gets 3 rows, one to freeze each of its coordinates
            return indices.size() * 3;
        }

        void VertexPinConstraint::ProjectConstraint(MeshPtr &mesh, GeomPtr &geom)
        {
            for (size_t i = 0; i < indices.size(); i++)
            {
                GCVertex v_i = mesh->vertex(indices[i]);
                geom->inputVertexPositions[v_i] = initPositions[i];
            }
        }

    } // namespace Constraints
} // namespace rsurfaces