#pragma once

#include "rsurface_types.h"
#include "block_cluster_tree.h"

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

    void addMetric(rsurfaces::Hs::HsMetric* hs_)
    {
        hs = hs_;
    }

    const rsurfaces::BlockClusterTree *getTree() const
    {
        return bct;
    }

    const rsurfaces::Hs::HsMetric *getHs()
    {
        return hs;
    }

    const Eigen::SparseMatrix<double> &getConstraintBlock() const
    {
        return *C;
    }

    void setEpsilon(double e)
    {
        epsilon = e;
    }

    double epsilon;

private:
    const rsurfaces::BlockClusterTree *bct;
    const Eigen::SparseMatrix<double> *C;
    const rsurfaces::Hs::HsMetric *hs;
};

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

                bct->MultiplyVector3Const(rhs, product, rsurfaces::BCTKernelType::HighOrder, true, lhs.epsilon);
                bct->MultiplyVector3Const(rhs, product, rsurfaces::BCTKernelType::LowOrder, true, lhs.epsilon);
                bct->MultiplyConstraintBlock(rhs, product, lhs.getConstraintBlock(), true);

                dst += product;
            }
        };

    } // namespace internal
} // namespace Eigen
