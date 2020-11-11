#include "sobolev/constraints/barycenter.h"
#include "helpers.h"

namespace rsurfaces
{
    namespace Constraints
    {
        void addSingleTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
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

        BarycenterConstraint3X::BarycenterConstraint3X(MeshPtr &mesh, GeomPtr &geom)
        {
            ResetFunction(mesh, geom);
        }

        void BarycenterConstraint3X::ResetFunction(MeshPtr &mesh, GeomPtr &geom)
        {
            initValue = meshBarycenter(geom, mesh);
        }

        void BarycenterConstraint3X::addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            // Take the same weights from the non-3X version of this constraint,
            // and duplicate them 3 times on each 3x3 diagonal block.
            std::vector<Triplet> singleTriplets;
            addSingleTriplets(singleTriplets, mesh, geom, 0);

            for (Triplet t : singleTriplets)
            {
                triplets.push_back(Triplet(baseRow + 3 * t.row(), 3 * t.col(), t.value()));
                triplets.push_back(Triplet(baseRow + 3 * t.row() + 1, 3 * t.col() + 1, t.value()));
                triplets.push_back(Triplet(baseRow + 3 * t.row() + 2, 3 * t.col() + 2, t.value()));
            }
        }

        void BarycenterConstraint3X::addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            std::vector<Triplet> singleTriplets;
            addSingleTriplets(singleTriplets, mesh, geom, 0);

            for (Triplet t : singleTriplets)
            {
                M(baseRow + 3 * t.row(), 3 * t.col()) = t.value();
                M(baseRow + 3 * t.row() + 1, 3 * t.col() + 1) = t.value();
                M(baseRow + 3 * t.row() + 2, 3 * t.col() + 2) = t.value();
            }
        }

        void BarycenterConstraint3X::addErrorValues(Eigen::VectorXd &V, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            Vector3 current = meshBarycenter(geom, mesh);
            V(baseRow    ) = current.x - initValue.x;
            V(baseRow + 1) = current.y - initValue.y;
            V(baseRow + 2) = current.z - initValue.z;
        }


        size_t BarycenterConstraint3X::nRows()
        {
            return 3;
        }

        void BarycenterConstraint3X::ProjectConstraint(MeshPtr &mesh, GeomPtr &geom)
        {
            Vector3 center = meshBarycenter(geom, mesh);
            translateMesh(geom, mesh, initValue - center);
        }

    } // namespace Constraints
} // namespace rsurfaces