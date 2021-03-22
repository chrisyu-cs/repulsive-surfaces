#include "sobolev/h2.h"
#include "sobolev/h1.h"

namespace rsurfaces
{
    namespace H2
    {
        void getTriplets(std::vector<Triplet> &triplets, const MeshPtr &mesh, const GeomPtr &geom, double epsilon)
        {
            size_t nRows = mesh->nVertices();
            Eigen::SparseMatrix<double> laplacian(nRows, nRows);
            Eigen::SparseMatrix<double> biLaplacian(nRows, nRows);
            std::vector<Triplet> lTriplets;

            // Build the Laplacian
            H1::getTriplets(lTriplets, mesh, geom, 0, false);
            laplacian.setFromTriplets(lTriplets.begin(), lTriplets.end());

            Eigen::VectorXd mass;
            mass.setZero(nRows);
            VertexIndices inds = mesh->getVertexIndices();
            for (GCVertex v : mesh->vertices())
            {
                double vMass = 1.0 / geom->vertexDualArea(v);
                mass(inds[v]) = vMass;
            }

            // Multiply it with itself
            biLaplacian = laplacian * mass.asDiagonal() * laplacian;

            // Add the triplets back to the vector
            for (int k = 0; k < biLaplacian.outerSize(); ++k)
            {
                for (SparseMatrix<double>::InnerIterator it(biLaplacian, k); it; ++it)
                {
                    // If on the diagonal, add the epsilon here
                    double eps = (it.row() == it.col()) ? epsilon : 0;
                    triplets.emplace_back(it.row(), it.col(), it.value() + eps);
                }
            }
        }
    } // namespace H2
} // namespace rsurfaces