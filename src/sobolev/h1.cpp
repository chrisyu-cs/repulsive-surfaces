#include "sobolev/h1.h"
#include "helpers.h"

namespace rsurfaces {
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
}