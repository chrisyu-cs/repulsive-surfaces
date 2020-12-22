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

        class HsMetric;

        template <typename HsPtr>
        void GetSchurComplement(const HsPtr hs, SchurComplement &dest);

        void ProjectViaSchurV(const HsMetric &hs, Eigen::VectorXd &gradient, Eigen::VectorXd &dest);

        /*
        * This class contains everything necessary to compute an Hs-projected
        * gradient direction. A new instance should be used every timestep.
        */
        class HsMetric
        {
        public:
            HsMetric(SurfaceEnergy *energy_);
            HsMetric(SurfaceEnergy *energy_, std::vector<Constraints::SimpleProjectorConstraint *> &spcs,
                     std::vector<ConstraintPack> &schurs);
            ~HsMetric();

            // Build the "high order" fractional Laplacian of order 2s.
            void FillMatrixHigh(Eigen::MatrixXd &M, double s, const MeshPtr &mesh, const GeomPtr &geom) const;
            // Add the regularizing "low order" term.
            void FillMatrixLow(Eigen::MatrixXd &M, double s, const MeshPtr &mesh, const GeomPtr &geom) const;

            // Build the base fractional Laplacian of order s.
            void FillMatrixFracOnly(Eigen::MatrixXd &M, double s, const MeshPtr &mesh, const GeomPtr &geom) const;
            // Build an exact Hs preconditioner with high- and low-order terms.
            Eigen::MatrixXd GetHsMatrixConstrained() const;

            // Build just the constraint block of the saddle matrix.
            Eigen::SparseMatrix<double> GetConstraintBlock(bool includeNewton = true) const;

            void ProjectGradientExact(const Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest) const;

            void ProjectSimpleConstraints();
            void ProjectSimpleConstraintsWithSaddle();

            template <typename Rhs>
            inline Rhs InvertMetricTemplated(const Rhs &gradient) const
            {
                Eigen::VectorXd temp = gradient;
                ProjectSparse(temp, temp);
                return Rhs(temp);
            }

            template <typename Rhs>
            inline Rhs InvertMetricSchurTemplated(const Rhs &gradient) const
            {
                Eigen::VectorXd temp = gradient;
                ProjectViaSchurV(*this, temp, temp);
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

            inline size_t getNumConstraints() const
            {
                size_t nConstraints = 0;

                for (Constraints::SimpleProjectorConstraint *cons : simpleConstraints)
                {
                    nConstraints += cons->nRows();
                }

                for (const ConstraintPack &schur : newtonConstraints)
                {
                    nConstraints += schur.constraint->nRows();
                }

                return nConstraints;
            }

            inline size_t getNumRows() const
            {
                return 3 * mesh->nVertices() + getNumConstraints();
            }

            inline BVHNode6D *GetBVH() const
            {
                return bvh;
            }

            inline double getBHTheta() const
            {
                return bh_theta;
            }

            inline void SetNewtonConstraints(std::vector<ConstraintPack> &newConstrs)
            {
                newtonConstraints.clear();
                newtonConstraints = newConstrs;
            }

            inline void SetSimpleConstraints(std::vector<Constraints::SimpleProjectorConstraint *> &simples)
            {
                simpleConstraints.clear();
                simpleConstraints = simples;
            }

            size_t topLeftNumRows() const;
            MeshPtr mesh;
            GeomPtr geom;

            std::vector<Constraints::SimpleProjectorConstraint *> simpleConstraints;
            std::vector<ConstraintPack> newtonConstraints;

            inline SchurComplement &Schur() const
            {
                if (!schurComplementComputed)
                {
                    GetSchurComplement(this, schurComplement);
                    schurComplementComputed = true;
                }
                return schurComplement;
            }

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
            bool usedDefaultConstraint;

            mutable SparseFactorization factorizedLaplacian;
            mutable BlockClusterTree *bct;
            mutable bool schurComplementComputed;
            mutable SchurComplement schurComplement;
        };


        template <typename HsPtr>
        void GetSchurComplement(const HsPtr hs, SchurComplement &dest)
        {
            size_t nVerts = hs->mesh->nVertices();
            size_t compNRows = 0;
            size_t bigNRows = hs->topLeftNumRows();

            // Figure out how many rows the constraint block is
            for (const ConstraintPack &c : hs->newtonConstraints)
            {
                compNRows += c.constraint->nRows();
            }
            if (compNRows == 0)
            {
                std::cout << "No constraints provided to Schur complement." << std::endl;
                throw 1;
            }

            dest.C.setZero(compNRows, bigNRows);
            size_t curRow = 0;

            // Fill in the constraint block by getting the entries for each constraint
            // while incrementing the rows
            for (const ConstraintPack &c : hs->newtonConstraints)
            {
                c.constraint->addEntries(dest.C, hs->mesh, hs->geom, curRow);
                curRow += c.constraint->nRows();
            }

            // https://en.wikipedia.org/wiki/Schur_complement
            // We want to compute (M/A) = D - C A^{-1} B.
            // In our case, D = 0, and B = C^T, so this is C A^{-1} C^T.
            // Unfortunately this means we have to apply A^{-1} once for each column of C^T,
            // which could get expensive if we have too many constraints.

            // First allocate some space for a single column
            Eigen::VectorXd curCol;
            curCol.setZero(bigNRows);
            // And some space for A^{-1} C^T
            Eigen::MatrixXd A_inv_CT;
            A_inv_CT.setZero(bigNRows, compNRows);

            // For each column, copy it into curCol, and do the solve for A^{-1}
            for (size_t r = 0; r < compNRows; r++)
            {
                // Copy the row of C into the column
                for (size_t i = 0; i < 3 * nVerts; i++)
                {
                    curCol(i) = dest.C(r, i);
                }
                hs->InvertMetric(curCol, curCol);
                // Copy the column into the column of A^{-1} C^T
                for (size_t i = 0; i < bigNRows; i++)
                {
                    A_inv_CT(i, r) = curCol(i);
                }
            }

            // Now we've multiplied A^{-1} C^T, so just multiply this with C and negate it
            dest.M_A = -dest.C * A_inv_CT;
        }


    } // namespace Hs

} // namespace rsurfaces
