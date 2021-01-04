#pragma once

#include "sobolev/hs.h"
#include "bct_matrix_replacement.h"

#include <unsupported/Eigen/IterativeSolvers>

namespace rsurfaces
{
    namespace Hs
    {
        void ProjectConstrainedHsIterativeMat(Hs::HsMetric &hs, Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest);
        void ProjectConstrainedHsIterative(Hs::HsMetric &hs, Eigen::VectorXd &gradient, Eigen::VectorXd &dest);

        template <typename V, typename Dest>
        void ProjectUnconstrainedHsIterativeCG(const Hs::HsMetric &hs, const V &gradient, Dest &dest,
                                             Eigen::SparseMatrix<double> &constraintBlock)
        {
            BlockClusterTree *bct = new BlockClusterTree(hs.mesh, hs.geom, hs.GetBVH(), hs.getBHTheta(), hs.getHsOrder());

            BCTMatrixReplacement fracL;
            fracL.addTree(bct);
            fracL.addConstraintBlock(constraintBlock);
            fracL.addMetric(&hs);

            bct->recomputeBarycenters();
            bct->PremultiplyAf1(BCTKernelType::HighOrder);
            bct->PremultiplyAf1(BCTKernelType::LowOrder);

            Eigen::ConjugateGradient<BCTMatrixReplacement, Eigen::Lower | Eigen::Upper, SparseHsPreconditioner> cg;
            // Eigen::GMRES<BCTMatrixReplacement, SparseHsPreconditioner> cg;
            cg.compute(fracL);
            
            Eigen::VectorXd temp;
            temp.setZero(gradient.rows());
            cg.setTolerance(1e-1);
            cg.setMaxIterations(20);
            temp = cg.solveWithGuess(gradient, temp);
            std::cout << "  * Converged in " << cg.iterations() << " iterations, final residual = " << cg.error() << std::endl;

            dest = temp;

            delete bct;
        }

        template <typename V, typename Dest>
        void ProjectUnconstrainedHsIterativeGMRES(const Hs::HsMetric &hs, const V &gradient, Dest &dest,
                                             Eigen::SparseMatrix<double> &constraintBlock)
        {
            BlockClusterTree *bct = new BlockClusterTree(hs.mesh, hs.geom, hs.GetBVH(), hs.getBHTheta(), hs.getHsOrder());

            BCTMatrixReplacement fracL;
            fracL.addTree(bct);
            fracL.addConstraintBlock(constraintBlock);
            fracL.addMetric(&hs);

            bct->recomputeBarycenters();
            bct->PremultiplyAf1(BCTKernelType::HighOrder);
            bct->PremultiplyAf1(BCTKernelType::LowOrder);

            Eigen::GMRES<BCTMatrixReplacement, SparseHsPreconditioner> cg;
            cg.compute(fracL);
            
            Eigen::VectorXd temp;
            temp.setZero(gradient.rows());
            cg.setTolerance(1e-4);
            cg.setMaxIterations(20);
            temp = cg.solveWithGuess(gradient, temp);
            std::cout << "  * Converged in " << cg.iterations() << " iterations, final residual = " << cg.error() << std::endl;

            dest = temp;

            delete bct;
        }


        template <typename V, typename Dest>
        void ProjectUnconstrainedHsIterative(const Hs::HsMetric &hs, const V &gradient, Dest &dest,
                                             Eigen::SparseMatrix<double> &constraintBlock)
        {
            BlockClusterTree *bct = new BlockClusterTree(hs.mesh, hs.geom, hs.GetBVH(), hs.getBHTheta(), hs.getHsOrder());

            BCTMatrixReplacement fracL;
            fracL.addTree(bct);
            fracL.addConstraintBlock(constraintBlock);
            fracL.addMetric(&hs);

            bct->recomputeBarycenters();
            bct->PremultiplyAf1(BCTKernelType::HighOrder);
            bct->PremultiplyAf1(BCTKernelType::LowOrder);

            // Eigen::ConjugateGradient<BCTMatrixReplacement, Eigen::Lower | Eigen::Upper, SparseHsPreconditioner> cg;
            Eigen::GMRES<BCTMatrixReplacement, SparseHsPreconditioner> cg;
            cg.compute(fracL);
            

            Eigen::VectorXd temp;
            temp.setZero(gradient.rows());
            cg.setTolerance(1e-4);
            cg.setMaxIterations(20);
            temp = cg.solveWithGuess(gradient, temp);
            std::cout << "  * Converged in " << cg.iterations() << " iterations, final residual = " << cg.error() << std::endl;

            dest = temp;

            delete bct;
        }

        template <typename V, typename Dest>
        void ProjectUnconstrainedHsIterative(const Hs::HsMetric &hs, const V &gradient, Dest &dest)
        {
            Eigen::SparseMatrix<double> constraintBlock = hs.GetConstraintBlock(false);
            ProjectUnconstrainedHsIterative(hs, gradient, dest, constraintBlock);
        }

        class IterativeInverse
        {
        public:
            template <typename V, typename Dest>
            static void Apply(const HsMetric &hs, const V &gradient, Dest &dest)
            {
                ProjectUnconstrainedHsIterative(hs, gradient, dest);
            }
        };
    } // namespace Hs
} // namespace rsurfaces
