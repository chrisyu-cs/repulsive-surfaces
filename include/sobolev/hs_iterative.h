#pragma once

#include "sobolev/hs.h"
#include "bct_matrix_replacement.h"

namespace rsurfaces
{
    namespace Hs
    {
        void ProjectConstrainedHsIterative(Hs::HsMetric &hs, Eigen::VectorXd &gradient, Eigen::VectorXd &dest);

        template <typename V, typename Dest>
        void ProjectUnconstrainedHsIterative(const Hs::HsMetric &hs, const V &gradient, Dest &dest,
                                             Eigen::SparseMatrix<double> &constraintBlock)
        {
            BlockClusterTree *bct = new BlockClusterTree(hs.mesh, hs.geom, hs.GetBVH(), hs.getBHTheta(), hs.getHsOrder());

            BCTMatrixReplacement fracL;
            fracL.addTree(bct);
            fracL.addConstraintBlock(constraintBlock);
            fracL.addMetric(&hs);

            std::cout << "Matrix-vector product expects " << fracL.rows() << " rows" << std::endl;

            bct->recomputeBarycenters();
            bct->PremultiplyAf1(BCTKernelType::HighOrder);
            bct->PremultiplyAf1(BCTKernelType::LowOrder);

            Eigen::ConjugateGradient<BCTMatrixReplacement, Eigen::Lower | Eigen::Upper, SparseHsPreconditioner> cg;
            cg.compute(fracL);

            dest.setZero();
            Eigen::VectorXd initialGuess = dest;

            size_t nRows = hs.topLeftNumRows();
            std::cout << "Using sparse metric only for initial guess" << std::endl;

            hs.InvertMetric(gradient, dest);

            dest.setZero();
            cg.setTolerance(1e-2);
            dest = cg.solveWithGuess(gradient, dest);
            std::cout << "CG num iterations: " << cg.iterations() << ", estimated error: " << cg.error() << std::endl;

            delete bct;
        }

        template <typename V, typename Dest>
        void ProjectUnconstrainedHsIterative(const Hs::HsMetric &hs, const V &gradient, Dest &dest)
        {
            Eigen::SparseMatrix<double> constraintBlock = hs.GetConstraintBlock(false);
            std::cout << "Constraint block has " << constraintBlock.rows() << " rows" << std::endl;
            ProjectUnconstrainedHsIterative(hs, gradient, dest, constraintBlock);
        }

        class IterativeInverse
        {
        public:
            template <typename V, typename Dest>
            void Apply(const HsMetric &hs, const V &gradient, Dest &dest) const
            {
                ProjectUnconstrainedHsIterative(hs, gradient, dest);
            }
        };
    } // namespace Hs
} // namespace rsurfaces
