#pragma once

#include "rsurface_types.h"
#include "matrix_utils.h"

namespace rsurfaces
{

    namespace Hs
    {
        Vector3 HatGradientOnTriangle(GCFace face, GCVertex vert, GeomPtr &geom);
        double get_s(double alpha, double beta);
        // Build the "high order" fractional Laplacian of order 2s.
        void FillMatrixHigh(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);
        // Build the base fractional Laplacian of order s.
        void FillMatrixFracOnly(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);

        void ProjectGradient(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom);
        void ProjectViaConvolution(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom);
        void ProjectViaSparse(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom);

    } // namespace Hs

} // namespace rsurfaces
