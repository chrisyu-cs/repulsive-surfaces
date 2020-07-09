#pragma once

#include "rsurface_types.h"
#include "matrix_utils.h"

namespace rsurfaces
{

    namespace Hs
    {
        Vector3 HatGradientOnTriangle(GCFace face, GCVertex vert, GeomPtr &geom);
        void AddTriangleContribution(Eigen::MatrixXd &M, double s, GCFace f1, GCFace f2, GeomPtr &geom, VertexIndices &indices);
        double get_s(double alpha, double beta);
        void FillMatrix(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);

        void ProjectGradient(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom);
        void ProjectViaConvolution(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom);

    } // namespace Hs

} // namespace rsurfaces
