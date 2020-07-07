#pragma once

#include "rsurface_types.h"
#include "matrix_utils.h"

namespace rsurfaces
{

    namespace Hs
    {
        inline double MetricDistanceTerm(double s, Vector3 v1, Vector3 v2)
        {
            double dist_term = 1.0 / pow(norm2(v1 - v2), s - 1);
            return dist_term;
        }

        Vector3 HatGradientOnTriangle(GCFace face, GCVertex vert, GeomPtr &geom);
        void AddTriangleContribution(Eigen::MatrixXd &M, double s, GCFace f1, GCFace f2, GeomPtr &geom, VertexIndices &indices);
        double get_s(double alpha, double beta);
        void FillMatrix(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom);
        inline void FillMatrix(Eigen::MatrixXd &M, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom)
        {
            FillMatrix(M, get_s(alpha, beta), mesh, geom);
        }

    } // namespace Hs

} // namespace rsurfaces
