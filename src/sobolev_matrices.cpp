#include "sobolev_matrices.h"

namespace rsurfaces
{
    GCVertex getOppositeVertex(GCEdge &e, GCVertex &v)
    {
        GCHalfedge he = e.halfedge();
        if (he.vertex() == v)
        {
            return he.twin().vertex();
        }
        else
        {
            return he.vertex();
        }
    }

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
            }
        }
    } // namespace Constraints

    namespace H1
    {
        void getTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom)
        {
            geom->requireEdgeCotanWeights();
            geom->requireVertexDualAreas();
            VertexIndices indices = mesh->getVertexIndices();
            for (GCVertex v : mesh->vertices())
            {
                double rowSum = 0;
                double area = geom->vertexDualAreas[v];

                for (GCEdge e : v.adjacentEdges())
                {
                    GCVertex opp = getOppositeVertex(e, v);
                    double wt = geom->edgeCotanWeight(e);
                    rowSum += wt;
                    triplets.push_back(Triplet(indices[v], indices[opp], -wt / area));
                }
                triplets.push_back(Triplet(indices[v], indices[v], rowSum / area));
            }
            geom->unrequireVertexDualAreas();
            geom->unrequireEdgeCotanWeights();
        }
    } // namespace H1
} // namespace rsurfaces