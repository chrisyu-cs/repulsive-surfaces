#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
    typedef Eigen::Triplet<double> Triplet;

    namespace MatrixUtils
    {
        void TripleTriplets(std::vector<Triplet> &orig, std::vector<Triplet> &output);
        void MatrixIntoColumn(Eigen::MatrixXd &M, Eigen::VectorXd &out);
        void ColumnIntoMatrix(Eigen::VectorXd &v, Eigen::MatrixXd &out);

        void SolveSparseSystem(Eigen::SparseMatrix<double> &M, Eigen::VectorXd &rhs, Eigen::VectorXd &output);
    } // namespace MatrixUtils
}