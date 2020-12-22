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
            typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;

        public:
            typedef typename Vector::StorageIndex StorageIndex;
            enum
            {
                ColsAtCompileTime = Eigen::Dynamic,
                MaxColsAtCompileTime = Eigen::Dynamic
            };
            SparseHsPreconditioner() {}

            template <typename MatrixType>
            explicit SparseHsPreconditioner(const MatrixType &fracL)
            {
                compute(fracL);
            }

            Eigen::Index rows() const { return hs->getNumRows(schurConstraints); }
            Eigen::Index cols() const { return hs->getNumRows(schurConstraints); }

            template <typename MatrixType>
            SparseHsPreconditioner &analyzePattern(const MatrixType &) { return *this; }

            template <typename MatrixType>
            SparseHsPreconditioner &factorize(const MatrixType &) { return *this; }

            template <typename MatrixType>
            SparseHsPreconditioner &compute(const MatrixType &fracL)
            {
                hs = fracL.getHs();
                std::cout << "Set Hs (" << hs->getNumRows(schurConstraints) << " rows)" << std::endl;
                return *this;
            }

            /** \internal */
            template <typename Rhs, typename Dest>
            void _solve_impl(const Rhs &b, Dest &x) const
            {
                x = hs->InvertMetricTemplated(b);
                std::cout << "Solution norm = " << x.norm() << std::endl;
            }

            template <typename Rhs>
            inline const Eigen::Solve<SparseHsPreconditioner, Rhs>
            solve(const Eigen::MatrixBase<Rhs> &b) const
            {
                eigen_assert(m_invdiag.size() == b.rows() && "DiagonalPreconditioner::solve(): invalid number of rows of the right hand side matrix b");
                return Eigen::Solve<SparseHsPreconditioner, Rhs>(*this, b.derived());
            }

            const Hs::HsMetric *hs;
            const std::vector<ConstraintPack> schurConstraints;

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
            cg.setTolerance(1e-2);
            dest = cg.solveWithGuess(gradient, initialGuess);
            std::cout << "CG num iterations: " << cg.iterations() << ", estimated error: " << cg.error() << std::endl;

            delete bct;
        }

    } // namespace Hs
} // namespace rsurfaces
