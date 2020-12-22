#include "sobolev/hs_iterative.h"
#include "sobolev/hs_schur.h"

namespace rsurfaces
{
    namespace Hs
    {
        void ProjectUnconstrainedHsIterative(Hs::HsMetric &hs, Eigen::VectorXd &gradient, Eigen::VectorXd &dest, bool includeNewton)
        {
            Eigen::SparseMatrix<double> constraintBlock = hs.GetConstraintBlock(includeNewton);
            std::cout << "Constraint block has " << constraintBlock.rows() << " rows" << std::endl;
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

            if (hs.newtonConstraints.size() > 0 && includeNewton)
            {
                std::cout << "Using Schur complement for initial guess" << std::endl;
                ProjectViaSchurV(hs, gradient, initialGuess);
            }
            else
            {
                std::cout << "Using sparse metric only for initial guess" << std::endl;
                hs.InvertMetric(gradient, initialGuess);
            }

            dest.setZero();
            cg.setTolerance(1e-2);
            dest = cg.solveWithGuess(gradient, dest);
            std::cout << "CG num iterations: " << cg.iterations() << ", estimated error: " << cg.error() << std::endl;

            delete bct;
        }

    } // namespace Hs
} // namespace rsurfaces
