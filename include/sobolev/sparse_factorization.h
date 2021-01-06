#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
    struct SparseFactorization
    {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;
        size_t nRows = 0;
        bool initialized = false;

        inline void Compute(Eigen::SparseMatrix<double> M)
        {
            nRows = M.rows();
            initialized = true;
            factor.compute(M);
        }

        inline Eigen::VectorXd Solve(const Eigen::VectorXd &v)
        {
            if (!initialized)
            {
                std::cerr << "Sparse factorization was not initialized before attempting to solve." << std::endl;
                throw 1;
            }
            return factor.solve(v);
        }

        inline Eigen::VectorXd SolveWithMasses(const Eigen::VectorXd &v, Eigen::VectorXd &mass)
        {
            if (!initialized)
            {
                std::cerr << "Sparse factorization was not initialized before attempting to solve." << std::endl;
                throw 1;
            }
            // Eigen::VectorXd
            return factor.solve(v);
        }
    };
} // namespace rsurfaces