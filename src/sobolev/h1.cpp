#include "sobolev/h1.h"
#include "helpers.h"
#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace H1
    {
        void getTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, bool premultiplyMass)
        {
            geom->requireEdgeCotanWeights();
            VertexIndices indices = mesh->getVertexIndices();
            for (GCVertex v : mesh->vertices())
            {
                double rowSum = 0;
                double area = geom->vertexDualAreas[v];
                double mass = premultiplyMass ? geom->vertexDualAreas[v] : 1;

                for (GCEdge e : v.adjacentEdges())
                {
                    GCVertex opp = getOppositeVertex(e, v);
                    double wt = geom->edgeCotanWeight(e);
                    rowSum += wt;
                    triplets.push_back(Triplet(indices[v], indices[opp], -wt / mass));
                }
                triplets.push_back(Triplet(indices[v], indices[v], rowSum / mass + 1e-10));
            }
            geom->unrequireEdgeCotanWeights();
        }

        void ProjectGradient(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, MeshPtr &mesh, GeomPtr &geom, bool useMass)
        {
            // Assemble the metric matrix
            std::vector<Triplet> triplets, triplets3x;
            H1::getTriplets(triplets, mesh, geom, false);
            Constraints::BarycenterConstraint bconstraint;
            Constraints::addTripletsToSymmetric(bconstraint, triplets, mesh, geom, mesh->nVertices());
            // Reduplicate the entries 3x along diagonals
            MatrixUtils::TripleTriplets(triplets, triplets3x);
            Eigen::SparseMatrix<double> metric(3 * mesh->nVertices() + 3, 3 * mesh->nVertices() + 3);
            metric.setFromTriplets(triplets3x.begin(), triplets3x.end());

            dest = gradient;
            if (useMass)
            {
                MultiplyByMass(dest, mesh, geom);
            }

            // Flatten the gradient into a single column
            Eigen::VectorXd gradientCol;
            gradientCol.setZero(3 * mesh->nVertices() + 3);
            MatrixUtils::MatrixIntoColumn(dest, gradientCol);

            // Invert the metric, and write it into the destination
            MatrixUtils::SolveSparseSystem(metric, gradientCol, gradientCol);
            MatrixUtils::ColumnIntoMatrix(gradientCol, dest);
        }
    } // namespace H1
} // namespace rsurfaces