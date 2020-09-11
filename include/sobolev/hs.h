#pragma once

#include "rsurface_types.h"
#include "matrix_utils.h"
#include "constraints.h"
#include "block_cluster_tree.h"
#include "hs_operators.h"

namespace rsurfaces
{
    using Constraints::ConstraintBase;

    namespace Hs
    {
        Vector3 HatGradientOnTriangle(GCFace face, GCVertex vert, GeomPtr &geom);
        double get_s(double alpha, double beta);
        // Build the "high order" fractional Laplacian of order 2s.
        void FillMatrixHigh(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);
        // Build the base fractional Laplacian of order s.
        void FillMatrixFracOnly(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);

        void ProjectGradient(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom);

        void ProjectViaSparseMat(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, SurfaceEnergy* energy, BlockClusterTree *&bct);

        void ProjectViaSparse(Eigen::VectorXd &gradient, Eigen::VectorXd &dest, SurfaceEnergy* energy, BlockClusterTree *&bct);

        struct SchurComplement
        {
            Eigen::MatrixXd C;
            Eigen::MatrixXd M_A;
        };

        void GetSchurComplement(std::vector<ConstraintBase *> constraints, SurfaceEnergy *energy, SchurComplement &dest, BlockClusterTree *&bct);

        void ProjectViaSchur(SchurComplement &comp, Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, SurfaceEnergy* energy, BlockClusterTree *&bct);

        void BackprojectViaSchur(std::vector<ConstraintBase *> constraints, SchurComplement &comp, SurfaceEnergy* energy, BlockClusterTree *&bct);

    } // namespace Hs

} // namespace rsurfaces
