#include "metric_term.h"
#include "matrix_utils.h"
#include "sobolev/h2.h"

namespace rsurfaces
{
    BiLaplacianMetricTerm::BiLaplacianMetricTerm(MeshPtr &mesh, GeomPtr &geom)
    {
        nMultiplyRows = 3 * mesh->nVertices();
        std::vector<Triplet> biTriplets, biTriplets3x;
        // Build the bi-Laplacian
        H2::getTriplets(biTriplets, mesh, geom, 1e-10);
        // Triple it to operate on vectors of length 3V
        MatrixUtils::TripleTriplets(biTriplets, biTriplets3x);

        biLaplacian.resize(nMultiplyRows, nMultiplyRows);
        biLaplacian.setFromTriplets(biTriplets3x.begin(), biTriplets3x.end());
    }

    void BiLaplacianMetricTerm::MultiplyAdd(Eigen::VectorXd &vec, Eigen::VectorXd &result) const
    {
        result.head(nMultiplyRows) += biLaplacian * vec.head(nMultiplyRows);
    }

}

