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

        inline double MetricDistanceTerm(double s, Vector3 v1, Vector3 v2)
        {
            double dist_term = 1.0 / pow(norm(v1 - v2), 2 * (s - 1) + 2);
            return dist_term;
        }

        inline double MetricDistanceTermFrac(double s, Vector3 v1, Vector3 v2)
        {
            double dist_term = 1.0 / pow(norm(v1 - v2), 2 * s + 2);
            return dist_term;
        }

        template <typename V, typename VF>
        void ApplyMidOperator(const MeshPtr &mesh, const GeomPtr &geom, V &a, VF &out)
        {
            FaceIndices fInds = mesh->getFaceIndices();
            VertexIndices vInds = mesh->getVertexIndices();

            for (GCFace face : mesh->faces())
            {
                double avg = 0;
                // Get one-third the value on all adjacent vertices
                for (GCVertex vert : face.adjacentVertices())
                {
                    avg += a(vInds[vert]) / 3.0;
                }
                out(fInds[face]) += avg;
            }
        }

        template <typename V, typename VF>
        void ApplyMidOperatorTranspose(const MeshPtr &mesh, const GeomPtr &geom, VF &a, V &out)
        {

            FaceIndices fInds = mesh->getFaceIndices();
            VertexIndices vInds = mesh->getVertexIndices();

            for (GCVertex vert : mesh->vertices())
            {
                double total = 0;
                // Put weight = 1/3 on all adjacent faces
                for (GCFace face : vert.adjacentFaces())
                {
                    total += a(fInds[face]) / 3.0;
                }
                out(vInds[vert]) += total;
            }
        }

    } // namespace Hs

} // namespace rsurfaces
