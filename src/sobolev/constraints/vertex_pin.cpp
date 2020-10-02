#include "sobolev/constraints/vertex_pin.h"

namespace rsurfaces
{
    namespace Constraints
    {
        VertexPinConstraint::VertexPinConstraint(MeshPtr &mesh, GeomPtr &geom, std::vector<size_t> indices_)
        {
            for (size_t i : indices_)
            {
                indices.push_back(i);
                initPositions.push_back(geom->inputVertexPositions[mesh->vertex(i)]);
            }
        }

        void VertexPinConstraint::addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            // All we do is put a 1 in the index for all pinned vertices
            for (size_t i = 0; i < indices.size(); i++)
            {
                triplets.push_back(Triplet(baseRow + i, indices[i], 1));
            }
        }

        void VertexPinConstraint::addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            // All we do is put a 1 in the index for all pinned vertices
            for (size_t i = 0; i < indices.size(); i++)
            {
                M(baseRow + i, indices[i]) = 1;
            }
        }

        void VertexPinConstraint::addValue(Eigen::VectorXd &V, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            std::cerr << "Can't backproject vertex pins." << std::endl;
            throw 1;
        }

        double VertexPinConstraint::getTargetValue()
        {
            return 0;
        }

        void VertexPinConstraint::incrementTargetValue(double incr)
        {
            std::cerr << "Can't increment vertex pins." << std::endl;
            throw 1;
        }

        size_t VertexPinConstraint::nRows()
        {
            return indices.size();
        }
    } // namespace Constraints
} // namespace rsurfaces