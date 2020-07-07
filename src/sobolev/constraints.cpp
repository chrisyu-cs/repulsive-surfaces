#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace Constraints
    {
        void addBarycenterTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
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
                triplets.push_back(Triplet(baseRow, indices[v], wt));
                triplets.push_back(Triplet(indices[v], baseRow, wt));
            }
        }

        void addBarycenterEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
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
                M(indices[v], baseRow) = wt;
            }
        }
    } // namespace Constraints
} // namespace rsurfaces