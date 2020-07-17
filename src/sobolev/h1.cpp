#include "sobolev/h1.h"
#include "helpers.h"
#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace H1
    {
        void getTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom)
        {
            geom->requireEdgeCotanWeights();
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
                    triplets.push_back(Triplet(indices[v], indices[opp], -wt));
                }
                triplets.push_back(Triplet(indices[v], indices[v], rowSum));
            }
            geom->unrequireEdgeCotanWeights();
        }

        void ProjectGradient(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, MeshPtr &mesh, GeomPtr &geom)
        {
            // Assemble the metric matrix
            std::vector<Triplet> triplets, triplets3x;
            H1::getTriplets(triplets, mesh, geom);
            Constraints::addBarycenterTriplets(triplets, mesh, geom, mesh->nVertices());
            // Reduplicate the entries 3x along diagonals
            MatrixUtils::TripleTriplets(triplets, triplets3x);
            Eigen::SparseMatrix<double> metric(3 * mesh->nVertices() + 3, 3 * mesh->nVertices() + 3);
            metric.setFromTriplets(triplets3x.begin(), triplets3x.end());

            // Flatten the gradient into a single column
            Eigen::VectorXd gradientCol;
            gradientCol.setZero(3 * mesh->nVertices() + 3);
            MatrixUtils::MatrixIntoColumn(gradient, gradientCol);

            // Invert the metric, and write it into the destination
            MatrixUtils::SolveSparseSystem(metric, gradientCol, gradientCol);
            MatrixUtils::ColumnIntoMatrix(gradientCol, dest);
        }
    } // namespace H1
} // namespace rsurfaces