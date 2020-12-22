#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
    typedef Eigen::Triplet<double> Triplet;

    namespace MatrixUtils
    {
        inline void SetRowFromVector3(Eigen::MatrixXd &M, int row, const Vector3 vec)
        {
            M(row, 0) = vec.x;
            M(row, 1) = vec.y;
            M(row, 2) = vec.z;
        }

        inline void addToRow(Eigen::MatrixXd &M, size_t row, const Vector3 v)
        {
            M(row, 0) += v.x;
            M(row, 1) += v.y;
            M(row, 2) += v.z;
        }

        inline Vector3 GetRowAsVector3(const Eigen::MatrixXd &M, int row)
        {
            return Vector3{M(row, 0), M(row, 1), M(row, 2)};
        }

        inline void GetTripletsFromSparse(const Eigen::SparseMatrix<double> &A, std::vector<Triplet> &triplets)
        {
            for (int k = 0; k < A.outerSize(); ++k)
            {
                for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
                {
                    int row = it.row();
                    int col = it.col();
                    double v = it.value();
                    triplets.push_back(Triplet{row, col, v});
                }
            }
        }

        void TripleTriplets(const std::vector<Triplet> &orig, std::vector<Triplet> &output);
        void TripleMatrix(const Eigen::MatrixXd &M, Eigen::MatrixXd &out);
        void MatrixIntoColumn(const Eigen::MatrixXd &M, Eigen::VectorXd &out);
        void ColumnIntoMatrix(const Eigen::VectorXd &v, Eigen::MatrixXd &out);

        void SolveSparseSystem(const Eigen::SparseMatrix<double> &M, Eigen::VectorXd &rhs, Eigen::VectorXd &output);
        void SolveDenseSystem(const Eigen::MatrixXd &M, Eigen::VectorXd &rhs, Eigen::VectorXd &output);
    } // namespace MatrixUtils
} // namespace rsurfaces