#pragma once

#include "rsurface_types.h"
#include "block_cluster_tree.h"
#include "sobolev/hs.h"

class BCTMatrixReplacement;
using Eigen::SparseMatrix;

// Adapted from https://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html

namespace Eigen
{
    namespace internal
    {
        // MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
        template <>
        struct traits<BCTMatrixReplacement> : public Eigen::internal::traits<Eigen::SparseMatrix<double>>
        {
        };
    } // namespace internal
} // namespace Eigen

class BCTMatrixReplacement : public Eigen::EigenBase<BCTMatrixReplacement>
{
public:
    // Required typedefs, constants, and method:
    typedef double Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    enum
    {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Eigen::Index rows() const
    {
        return bct->expectedNRows() + C->rows();
    }

    Eigen::Index outerSize() const
    {
        return bct->expectedNRows() + C->rows();
    }

    Eigen::Index innerSize() const
    {
        return bct->expectedNRows() + C->rows();
    }

    Eigen::Index cols() const
    {
        return bct->expectedNCols() + C->rows();
    }

    template <typename Rhs>
    Eigen::Product<BCTMatrixReplacement, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const
    {
        return Eigen::Product<BCTMatrixReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    // Custom API:
    BCTMatrixReplacement() {}

    void addTree(const rsurfaces::BlockClusterTree *bct_)
    {
        bct = bct_;
    }

    void addConstraintBlock(const Eigen::SparseMatrix<double> &C_)
    {
        C = &C_;
    }

    void addMetric(rsurfaces::Hs::HsMetric *hs_)
    {
        hs = hs_;
    }

    const rsurfaces::BlockClusterTree *getTree() const
    {
        return bct;
    }

    const rsurfaces::Hs::HsMetric *getHs() const
    {
        return hs;
    }

    const Eigen::SparseMatrix<double> &getConstraintBlock() const
    {
        return *C;
    }

private:
    const rsurfaces::BlockClusterTree *bct;
    const Eigen::SparseMatrix<double> *C;
    const rsurfaces::Hs::HsMetric *hs;
};

namespace rsurfaces
{

    namespace Hs
    {
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

            Eigen::Index rows() const { return hs->getNumRows(); }
            Eigen::Index cols() const { return hs->getNumRows(); }

            template <typename MatrixType>
            SparseHsPreconditioner &analyzePattern(const MatrixType &) { return *this; }

            template <typename MatrixType>
            SparseHsPreconditioner &factorize(const MatrixType &) { return *this; }

            template <typename MatrixType>
            SparseHsPreconditioner &compute(const MatrixType &fracL)
            {
                hs = fracL.getHs();
                std::cout << "Set Hs (" << hs->getNumRows() << " rows)" << std::endl;

                return *this;
            }

            /** \internal */
            template <typename Rhs, typename Dest>
            void _solve_impl(const Rhs &b, Dest &x) const
            {
                if (hs->newtonConstraints.size() > 0)
                {
                    x = hs->InvertMetricSchurTemplated(b);
                }
                else
                {
                    x = hs->InvertMetricTemplated(b);
                }
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
            SchurComplement schur;

            Eigen::ComputationInfo info() { return Eigen::Success; }
        };
    } // namespace Hs
} // namespace rsurfaces

namespace Eigen
{
    namespace internal
    {

        template <typename Rhs>
        struct generic_product_impl<BCTMatrixReplacement, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
            : generic_product_impl_base<BCTMatrixReplacement, Rhs, generic_product_impl<BCTMatrixReplacement, Rhs>>
        {
            typedef typename Product<BCTMatrixReplacement, Rhs>::Scalar Scalar;

            template <typename Dest>
            static void scaleAndAddTo(Dest &dst, const BCTMatrixReplacement &lhs, const Rhs &rhs, const Scalar &alpha)
            {
                // This method should implement "dst += alpha * lhs * rhs" inplace,
                // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
                assert(alpha == Scalar(1) && "scaling is not implemented");
                EIGEN_ONLY_USED_FOR_DEBUG(alpha);

                const rsurfaces::BlockClusterTree *bct = lhs.getTree();

                Eigen::VectorXd product(bct->expectedNRows() + lhs.getConstraintBlock().rows());
                product.setZero();

                bct->MultiplyVector3(rhs, product, rsurfaces::BCTKernelType::HighOrder, true);
                bct->MultiplyVector3(rhs, product, rsurfaces::BCTKernelType::LowOrder, true);
                bct->MultiplyConstraintBlock(rhs, product, lhs.getConstraintBlock(), true);

                dst += product;
            }
        };

    } // namespace internal
} // namespace Eigen
