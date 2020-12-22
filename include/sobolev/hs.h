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
            size_t nRows = 0;
            bool initialized = false;

            inline void Compute(Eigen::SparseMatrix<double> M)
            {
                nRows = M.rows();
                initialized = true;
                factor.compute(M);
            }

            inline Eigen::VectorXd Solve(const Eigen::VectorXd &v)
            {
                if (!initialized)
                {
                    std::cerr << "Sparse factorization was not initialized before attempting to solve." << std::endl;
                    throw 1;
                }
                return factor.solve(v);
            }

            inline Eigen::VectorXd SolveWithMasses(const Eigen::VectorXd &v, Eigen::VectorXd &mass)
            {
                if (!initialized)
                {
                    std::cerr << "Sparse factorization was not initialized before attempting to solve." << std::endl;
                    throw 1;
                }
                // Eigen::VectorXd
                return factor.solve(v);
            }
        };

        Vector3 HatGradientOnTriangle(GCFace face, GCVertex vert, GeomPtr &geom);
        double get_s(double alpha, double beta);

        /*
        * This class contains everything necessary to compute an Hs-projected
        * gradient direction. A new instance should be used every timestep.
        */
        class HsMetric
        {
        public:
            HsMetric(SurfaceEnergy *energy_);
            HsMetric(SurfaceEnergy *energy_, std::vector<Constraints::SimpleProjectorConstraint *> &spcs);
            ~HsMetric();

            // Build the "high order" fractional Laplacian of order 2s.
            void FillMatrixHigh(Eigen::MatrixXd &M, double s, const MeshPtr &mesh, const GeomPtr &geom) const;
            // Add the regularizing "low order" term.
            void FillMatrixLow(Eigen::MatrixXd &M, double s, const MeshPtr &mesh, const GeomPtr &geom) const;

            // Build the base fractional Laplacian of order s.
            void FillMatrixFracOnly(Eigen::MatrixXd &M, double s, const MeshPtr &mesh, const GeomPtr &geom) const;
            // Build an exact Hs preconditioner with high- and low-order terms.
            Eigen::MatrixXd GetHsMatrixConstrained(std::vector<ConstraintPack> &schurConstraints) const;

            // Build just the constraint block of the saddle matrix.
            Eigen::SparseMatrix<double> GetConstraintBlock(std::vector<ConstraintPack> &schurConstraints) const;

            void ProjectGradientExact(const Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, std::vector<ConstraintPack> &schurConstraints) const;

            void ProjectSimpleConstraints();
            void ProjectSimpleConstraintsWithSaddle();

            template <typename Rhs>
            inline Rhs InvertMetricTemplated(const Rhs &gradient) const
            {
                Eigen::VectorXd temp = gradient;
                ProjectSparse(temp, temp);
                return Rhs(temp);
            }

            inline void InvertMetric(const Eigen::VectorXd &gradient, Eigen::VectorXd &dest) const
            {
                ProjectSparse(gradient, dest);
            }
            
            inline Eigen::VectorXd InvertMetric(const Eigen::VectorXd &gradient) const
            {
                Eigen::VectorXd dest;
                dest.setZero(gradient.rows());
                ProjectSparse(gradient, dest);
                return dest;
            }

            inline void InvertMetricMat(const Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest) const
            {
                ProjectSparseMat(gradient, dest);
            }

            inline double getHsOrder() const
            {
                Vector2 exps = energy->GetExponents();
                return get_s(exps.x, exps.y);
            }

            inline size_t getNumConstraints(const std::vector<ConstraintPack> &schurConstraints) const
            {
                size_t nConstraints = 0;

                for (Constraints::SimpleProjectorConstraint *cons : simpleConstraints)
                {
                    nConstraints += cons->nRows();
                }

                for (const ConstraintPack &schur : schurConstraints)
                {
                    nConstraints += schur.constraint->nRows();
                }

                return nConstraints;
            }

            inline size_t getNumRows(const std::vector<ConstraintPack> &schurConstraints) const
            {
                return 3 * mesh->nVertices() + getNumConstraints(schurConstraints);
            }

            inline BVHNode6D *GetBVH() const
            {
                return bvh;
            }

            inline double getBHTheta() const
            {
                return bh_theta;
            }

            size_t topLeftNumRows() const;
            MeshPtr mesh;
            GeomPtr geom;

        private:
            void addSimpleConstraintEntries(Eigen::MatrixXd &M) const;
            void addSimpleConstraintTriplets(std::vector<Triplet> &triplets) const;
            void initFromEnergy(SurfaceEnergy *energy_);

            // Project the gradient into Hs by using the L^{-1} M L^{-1} factorization
            void ProjectSparse(const Eigen::VectorXd &gradient, Eigen::VectorXd &dest) const;
            // Same as above but with the input/output being matrices
            void ProjectSparseMat(const Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest) const;

            void ProjectSparseWithR1Update(const Eigen::VectorXd &gradient, Eigen::VectorXd &dest);
            void ProjectSparseWithR1UpdateMat(const Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest);

            BVHNode6D *bvh;
            double bh_theta;

            SurfaceEnergy *energy;
            std::vector<Constraints::SimpleProjectorConstraint *> simpleConstraints;
            std::vector<ConstraintPack> harderConstraints;
            bool usedDefaultConstraint;
            mutable SparseFactorization factorizedLaplacian;
            mutable BlockClusterTree *bct;
        };

    } // namespace Hs

} // namespace rsurfaces
