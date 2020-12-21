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

            inline Eigen::VectorXd Solve(Eigen::VectorXd &v)
            {
                if (!initialized)
                {
                    std::cerr << "Sparse factorization was not initialized before attempting to solve." << std::endl;
                    throw 1;
                }
                return factor.solve(v);
            }
            
            inline Eigen::VectorXd SolveWithMasses(Eigen::VectorXd &v, Eigen::VectorXd &mass)
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
            void FillMatrixHigh(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);
            // Add the regularizing "low order" term.
            void FillMatrixLow(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);
            
            // Build the base fractional Laplacian of order s.
            void FillMatrixFracOnly(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);
            // Build the base fractional Laplacian of order s.
            void FillMatrixVertsFirst(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);
            
            // Build an exact Hs preconditioner with high- and low-order terms.
            Eigen::MatrixXd GetHsMatrixConstrained(std::vector<ConstraintPack> &schurConstraints);

            // Build just the constraint block of the saddle matrix.
            Eigen::SparseMatrix<double> GetConstraintBlock(std::vector<ConstraintPack> &schurConstraints);

            void ProjectGradientExact(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, std::vector<ConstraintPack> &schurConstraints);

            void ProjectSimpleConstraints();
            void ProjectSimpleConstraintsWithSaddle();

            inline void InvertMetric(Eigen::VectorXd &gradient, Eigen::VectorXd &dest)
            {
                ProjectSparse(gradient, dest);
            }

            inline void InvertMetricMat(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest)
            {
                ProjectSparseMat(gradient, dest);
            }

            size_t topLeftNumRows();
            MeshPtr mesh;
            GeomPtr geom;

        private:
            void addSimpleConstraintEntries(Eigen::MatrixXd &M);
            void addSimpleConstraintTriplets(std::vector<Triplet> &triplets);
            void initFromEnergy(SurfaceEnergy *energy_);

            // Project the gradient into Hs by using the L^{-1} M L^{-1} factorization
            void ProjectSparse(Eigen::VectorXd &gradient, Eigen::VectorXd &dest);
            // Same as above but with the input/output being matrices
            void ProjectSparseMat(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest);

            void ProjectSparseWithR1Update(Eigen::VectorXd &gradient, Eigen::VectorXd &dest);
            void ProjectSparseWithR1UpdateMat(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest);

            BVHNode6D *bvh;
            double order_s;
            double bh_theta;

            SurfaceEnergy *energy;
            std::vector<Constraints::SimpleProjectorConstraint *> simpleConstraints;
            bool usedDefaultConstraint;
            SparseFactorization factorizedLaplacian;
            BlockClusterTree *bct;
        };

    } // namespace Hs

} // namespace rsurfaces
