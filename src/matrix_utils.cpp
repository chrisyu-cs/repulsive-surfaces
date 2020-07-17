#include "matrix_utils.h"
#include <Eigen/SparseCholesky>
#include <Eigen/LU>

namespace rsurfaces
{
    namespace MatrixUtils
    {
        void TripleTriplets(std::vector<Triplet> &orig, std::vector<Triplet> &output)
        {
            for (Triplet &t : orig)
            {
                output.push_back(Triplet(3 * t.row(), 3 * t.col(), t.value()));
                output.push_back(Triplet(3 * t.row() + 1, 3 * t.col() + 1, t.value()));
                output.push_back(Triplet(3 * t.row() + 2, 3 * t.col() + 2, t.value()));
            }
        }

        void TripleMatrix(Eigen::MatrixXd &M, Eigen::MatrixXd &out)
        {
            for (int i = 0; i < M.rows(); i++)
            {
                for (int j = 0; j < M.cols(); j++)
                {
                    out(3 * i, 3 * j) = M(i, j);
                    out(3 * i + 1, 3 * j + 1) = M(i, j);
                    out(3 * i + 2, 3 * j + 2) = M(i, j);
                }
            }
        }

        void MatrixIntoColumn(Eigen::MatrixXd &M, Eigen::VectorXd &out)
        {
            int rows = M.rows();
            int cols = M.cols();

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    int ind = cols * i + j;
                    out(ind) = M(i, j);
                }
            }
        }

        void ColumnIntoMatrix(Eigen::VectorXd &v, Eigen::MatrixXd &out)
        {
            int rows = out.rows();
            int cols = out.cols();

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    int ind = cols * i + j;
                    out(i, j) = v(ind);
                }
            }
        }

        void SolveSparseSystem(Eigen::SparseMatrix<double> &M, Eigen::VectorXd &rhs, Eigen::VectorXd &output)
        {
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(M);
            if (solver.info() != Eigen::Success)
            {
                // decomposition failed
                std::cout << "Sparse solve failed. Exiting." << std::endl;
                std::exit(1);
            }
            Eigen::VectorXd x = solver.solve(rhs);
            if (solver.info() != Eigen::Success)
            {
                // solving failed
                std::cout << "Sparse solve failed. Exiting." << std::endl;
                std::exit(1);
            }
            output = x;
        }

        void SolveDenseSystem(Eigen::MatrixXd &M, Eigen::VectorXd &rhs, Eigen::VectorXd &output)
        {
            Eigen::PartialPivLU<Eigen::MatrixXd> solver = M.partialPivLu();
            output = solver.solve(rhs);
        }

    } // namespace MatrixUtils
} // namespace rsurfaces