#include "sobolev/hs_iterative.h"

namespace rsurfaces
{
    namespace Hs
    {
        /** \ingroup IterativeLinearSolvers_Module
  * \brief A naive preconditioner which approximates any matrix as the identity matrix
  *
  * \implsparsesolverconcept
  *
  * \sa class DiagonalPreconditioner
  */
        class SparseHsPreconditioner
        {
        public:
            SparseHsPreconditioner() {}

            template <typename MatrixType>
            explicit SparseHsPreconditioner(const MatrixType &fracL)
            {
                hs = fracL.getHs();
            }

            template <typename MatrixType>
            SparseHsPreconditioner &analyzePattern(const MatrixType &) { return *this; }

            template <typename MatrixType>
            SparseHsPreconditioner &factorize(const MatrixType &) { return *this; }

            template <typename MatrixType>
            SparseHsPreconditioner &compute(const MatrixType &) { return *this; }

            template <typename Rhs>
            inline const Rhs &solve(const Rhs &b) const
            {
                return b;
            }

            const Hs::HsMetric* hs;

            Eigen::ComputationInfo info() { return Eigen::Success; }
        };

        void ProjectHsGradientIterative(Hs::HsMetric &hs, Eigen::VectorXd &gradient, Eigen::VectorXd &dest, std::vector<ConstraintPack> &schurConstraints)
        {
            Eigen::SparseMatrix<double> constraintBlock = hs.GetConstraintBlock(schurConstraints);
            BlockClusterTree *bct = new BlockClusterTree(hs.mesh, hs.geom, hs.GetBVH(), hs.getBHTheta(), hs.getHsOrder());

            BCTMatrixReplacement fracL;
            fracL.addTree(bct);
            fracL.addConstraintBlock(constraintBlock);
            fracL.setEpsilon(0);
            fracL.addMetric(&hs);

            bct->recomputeBarycenters();
            bct->PremultiplyAf1(BCTKernelType::HighOrder);
            bct->PremultiplyAf1(BCTKernelType::LowOrder);

            Eigen::ConjugateGradient<BCTMatrixReplacement, Eigen::Lower | Eigen::Upper, SparseHsPreconditioner> cg;
            cg.compute(fracL);

            dest.setZero();
            Eigen::VectorXd initialGuess = dest;
            hs.InvertMetric(gradient, initialGuess);

            dest.setZero();
            dest = cg.solveWithGuess(gradient, dest);
            std::cout << "CG num iterations: " << cg.iterations() << ", estimated error: " << cg.error() << std::endl;

            delete bct;
        }

    } // namespace Hs
} // namespace rsurfaces
