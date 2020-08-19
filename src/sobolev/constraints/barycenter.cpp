#include "sobolev/constraints/barycenter.h"

namespace rsurfaces
{
    namespace Constraints
    {
        void BarycenterConstraint::addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            // Just want to place normalized dual weights in the entry for each vertex
            geom->requireVertexDualAreas();
            VertexIndices indices = mesh->getVertexIndices();
            double sumArea = 0;
            for (GCVertex v : mesh->vertices())
            {
                sumArea += geom->vertexDualAreas[v];
            }

            for (GCVertex v : mesh->vertices())
            {
                double wt = geom->vertexDualAreas[v] / sumArea;
                triplets.push_back(Triplet(baseRow, indices[v], wt));
            }
        }

        void BarycenterConstraint::addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            geom->requireVertexDualAreas();
            VertexIndices indices = mesh->getVertexIndices();
            double sumArea = 0;
            for (GCVertex v : mesh->vertices())
            {
                sumArea += geom->vertexDualAreas[v];
            }

            for (GCVertex v : mesh->vertices())
            {
                double wt = geom->vertexDualAreas[v] / sumArea;
                M(baseRow, indices[v]) = wt;
            }
        }

        size_t BarycenterConstraint::nRows()
        {
            return 1;
        }

        void BarycenterConstraint3X::addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            // Take the same weights from the non-3X version of this constraint,
            // and duplicate them 3 times on each 3x3 diagonal block.
            BarycenterConstraint single;
            std::vector<Triplet> singleTriplets;
            single.addTriplets(singleTriplets, mesh, geom, 0);

            for (Triplet t : singleTriplets)
            {
                triplets.push_back(Triplet(baseRow + 3 * t.row(), 3 * t.col(), t.value()));
                triplets.push_back(Triplet(baseRow + 3 * t.row() + 1, 3 * t.col() + 1, t.value()));
                triplets.push_back(Triplet(baseRow + 3 * t.row() + 2, 3 * t.col() + 2, t.value()));
            }
        }

        void BarycenterConstraint3X::addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            BarycenterConstraint single;
            std::vector<Triplet> singleTriplets;
            single.addTriplets(singleTriplets, mesh, geom, 0);

            for (Triplet t : singleTriplets)
            {
                M(baseRow + 3 * t.row(), 3 * t.col()) = t.value();
                M(baseRow + 3 * t.row() + 1, 3 * t.col() + 1) = t.value();
                M(baseRow + 3 * t.row() + 2, 3 * t.col() + 2) = t.value();
            }
        }

        size_t BarycenterConstraint3X::nRows()
        {
            return 3;
        }

    } // namespace Constraints
} // namespace rsurfaces