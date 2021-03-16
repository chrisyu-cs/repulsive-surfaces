#pragma once

#include "rsurface_types.h"
#include "profiler.h"

namespace rsurfaces
{
    struct SparseFactorization
    {
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;
        size_t nRows = 0;
        bool initialized = false;

        inline void Compute(Eigen::SparseMatrix<double> M)
        {
            ptic("SparseFactorization::Compute");
            nRows = M.rows();
            initialized = true;
            factor.compute(M);
            ptoc("SparseFactorization::Compute");
        }

        inline Eigen::VectorXd Solve(const Eigen::VectorXd &v)
        {
            ptic("SparseFactorization::Solve");
            if (!initialized)
            {
                std::cerr << "Sparse factorization was not initialized before attempting to solve." << std::endl;
                throw 1;
            }
            auto result = factor.solve(v);
            ptoc("SparseFactorization::Solve");
            return result;
        }

        inline Eigen::VectorXd SolveWithMasses(const Eigen::VectorXd &v, Eigen::VectorXd &mass)
        {
            ptic("SparseFactorization::SolveWithMasses");
            if (!initialized)
            {
                std::cerr << "Sparse factorization was not initialized before attempting to solve." << std::endl;
                throw 1;
            }
            // Eigen::VectorXd
            
            auto result = factor.solve(v);
            ptoc("SparseFactorization::SolveWithMasses");
            return result;
        }
    };
} // namespace rsurfaces
