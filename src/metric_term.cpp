#include "metric_term.h"
#include "matrix_utils.h"
#include "sobolev/h1.h"

namespace rsurfaces
{
    BiLaplacianMetricTerm::BiLaplacianMetricTerm(MeshPtr &mesh, GeomPtr &geom)
    {
        nMultiplyRows = 3 * mesh->nVertices();

        Eigen::SparseMatrix<double> laplacian(nMultiplyRows, nMultiplyRows);
        std::vector<Triplet> lTriplets;

        // Build the Laplacian
        H1::getTriplets(lTriplets, mesh, geom, 0, false);
        laplacian.setFromTriplets(lTriplets.begin(), lTriplets.end());

        // Multiply it with itself
        biLaplacian = laplacian * laplacian;
    }

    void BiLaplacianMetricTerm::MultiplyAdd(Eigen::VectorXd &vec, Eigen::VectorXd &result) const
    {
        result.head(nMultiplyRows) += biLaplacian * vec.head(nMultiplyRows);
    }

}

