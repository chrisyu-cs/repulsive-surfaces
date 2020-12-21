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

    const rsurfaces::BlockClusterTree *getTree() const
    {
        return bct;
    }

    const Eigen::SparseMatrix<double> &getConstraintBlock() const
    {
        return *C;
    }

private:
    const rsurfaces::BlockClusterTree *bct;
    const Eigen::SparseMatrix<double> *C;
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

                bct->MultiplyVector3Const(rhs, dst, rsurfaces::BCTKernelType::HighOrder, true);
                bct->MultiplyVector3Const(rhs, dst, rsurfaces::BCTKernelType::LowOrder, true);
                bct->MultiplyConstraintBlock(rhs, dst, lhs.getConstraintBlock(), true);
            }
        };

    } // namespace internal
} // namespace Eigen