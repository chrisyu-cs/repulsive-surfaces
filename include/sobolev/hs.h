#pragma once

#include "rsurface_types.h"
#include "matrix_utils.h"
#include "constraints.h"
#include "block_cluster_tree.h"
#include "hs_operators.h"

#include <Eigen/Sparse>

namespace rsurfaces
{
    using Constraints::ConstraintBase;

    namespace Hs
    {
        struct SchurComplement
        {
            Eigen::MatrixXd C;
            Eigen::MatrixXd M_A;
        };

        struct SparseFactorization
        {
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;
            bool initialized = false;

            inline void Compute(Eigen::SparseMatrix<double> M)
            {
                initialized = true;
                factor.compute(M);
            }

            inline Eigen::VectorXd Solve(Eigen::VectorXd &v)
            {
                if (!initialized)
                {
                    std::cerr << "Sparse factorization was not initialized before attempting to solve." << std::endl;
                    throw 1;
                }
                return factor.solve(v);
            }
        };

        Vector3 HatGradientOnTriangle(GCFace face, GCVertex vert, GeomPtr &geom);
        double get_s(double alpha, double beta);
        // Build the "high order" fractional Laplacian of order 2s.
        void FillMatrixHigh(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);
        // Build the base fractional Laplacian of order s.
        void FillMatrixFracOnly(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);

        void ProjectGradient(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom);

        // Project the gradient into Hs by using the L^{-1} M L^{-1} factorization
        void ProjectViaSparse(Eigen::VectorXd &gradient, Eigen::VectorXd &dest, SurfaceEnergy *energy, BlockClusterTree *&bct, SparseFactorization &factor);
        // Same as above but with the input/output being matrices
        void ProjectViaSparseMat(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, SurfaceEnergy *energy, BlockClusterTree *&bct, SparseFactorization &factor);

        void GetSchurComplement(std::vector<ConstraintPack> constraints, SurfaceEnergy *energy, SchurComplement &dest,
                                BlockClusterTree *&bct, SparseFactorization &factor);

        void ProjectViaSchur(SchurComplement &comp, Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, SurfaceEnergy *energy,
                             BlockClusterTree *&bct, SparseFactorization &factor);

        void BackprojectViaSchur(std::vector<ConstraintPack> &constraints, SchurComplement &comp, SurfaceEnergy *energy,
                                 BlockClusterTree *&bct, SparseFactorization &factor);

    } // namespace Hs

} // namespace rsurfaces
