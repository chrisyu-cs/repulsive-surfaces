#pragma once

#include "matrix_utils.h"
#include "rsurface_types.h"
#include "energy/tpe_kernel.h"

namespace rsurfaces
{
    namespace Hs
    {

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

        inline double MetricDistanceTermLowPure(double s, Vector3 v1, Vector3 v2, Vector3 n1, Vector3 n2)
        {
            double sigma = s - 1;
            double s_pow = 2 * sigma + 1;
            double a = 2;
            double b = 4 + s_pow;
            return TPEKernel::tpe_Kf_symmetric(v1, v2, n1, n2, a, b);
        }

        // Return the 3x3 matrix representing the cross product with v.
        inline Eigen::Matrix3d CrossProductMatrix(Vector3 v)
        {
            Eigen::Matrix3d X;
            X.setZero();

            X(0, 1) = -v.z;
            X(0, 2) = v.y;
            X(1, 2) = -v.x;

            X(1, 0) = v.z;
            X(2, 0) = -v.y;
            X(2, 1) = v.x;
            return X;
        }

        inline void setCol3x3(Eigen::Matrix3d &M, Vector3 v, size_t c)
        {
            M(0, c) = v.x;
            M(1, c) = v.y;
            M(2, c) = v.z;
        }

        inline void addDfTriplets(std::vector<Triplet> &triplets, Eigen::Matrix3d &M_face, size_t faceIndex, size_t i1, size_t i2, size_t i3)
        {
            size_t faceRow = 3 * faceIndex;
            // First row of M_face goes into faceRow, with columns indexed by vertices
            triplets.push_back(Triplet(faceRow, i1, M_face(0, 0)));
            triplets.push_back(Triplet(faceRow, i2, M_face(0, 1)));
            triplets.push_back(Triplet(faceRow, i3, M_face(0, 2)));
            // Second row goes into faceRow + 1, same column indices
            triplets.push_back(Triplet(faceRow + 1, i1, M_face(1, 0)));
            triplets.push_back(Triplet(faceRow + 1, i2, M_face(1, 1)));
            triplets.push_back(Triplet(faceRow + 1, i3, M_face(1, 2)));
            // Third row goes into faceRow + 2, same column indices
            triplets.push_back(Triplet(faceRow + 2, i1, M_face(2, 0)));
            triplets.push_back(Triplet(faceRow + 2, i2, M_face(2, 1)));
            triplets.push_back(Triplet(faceRow + 2, i3, M_face(2, 2)));
        }

        inline Eigen::SparseMatrix<double> BuildDfOperator(const MeshPtr &mesh, const GeomPtr &geom)
        {
            FaceIndices fInds = mesh->getFaceIndices();
            VertexIndices vInds = mesh->getVertexIndices();

            std::vector<Triplet> triplets;

            for (GCFace face : mesh->faces())
            {
                if (face.isBoundaryLoop())
                {
                    continue;
                }

                // The expression for the gradient of a field g inside a triangle is
                // (1 / 2A) * N x (g_1 * e_23 + g_2 * g_31 + g_3 * g_12)
                Eigen::Matrix3d mat_f;
                mat_f.setZero();

                GCHalfedge he = face.halfedge();
                size_t i1 = vInds[he.vertex()];
                size_t i2 = vInds[he.next().vertex()];
                size_t i3 = vInds[he.next().next().vertex()];
                Vector3 p1 = geom->inputVertexPositions[he.vertex()];
                Vector3 p2 = geom->inputVertexPositions[he.next().vertex()];
                Vector3 p3 = geom->inputVertexPositions[he.next().next().vertex()];

                Vector3 e_12 = p2 - p1;
                Vector3 e_23 = p3 - p2;
                Vector3 e_31 = p1 - p3;

                // This ordering is so that, when multiplying by field values for
                // vertices (i, j, k), we get the gradient of the field
                setCol3x3(mat_f, e_23, 0);
                setCol3x3(mat_f, e_31, 1);
                setCol3x3(mat_f, e_12, 2);

                double area = geom->faceAreas[face];
                mat_f = CrossProductMatrix(geom->faceNormals[face]) * mat_f / (2 * area);

                addDfTriplets(triplets, mat_f, fInds[face], i1, i2, i3);
            }

            Eigen::SparseMatrix<double> Df(3 * mesh->nFaces(), mesh->nVertices());
            Df.setFromTriplets(triplets.begin(), triplets.end());
            return Df;
        }

        template <typename V, typename VF>
        void ApplyMidOperator(const MeshPtr &mesh, const GeomPtr &geom, V &a, VF &out)
        {
            FaceIndices fInds = mesh->getFaceIndices();
            VertexIndices vInds = mesh->getVertexIndices();

            for (GCFace face : mesh->faces())
            {
                if (face.isBoundaryLoop())
                {
                    continue;
                }
                double total = 0;
                // Get one-third the value on all adjacent vertices
                for (GCVertex vert : face.adjacentVertices())
                {
                    total += a(vInds[vert]) / 3.0;
                }
                out(fInds[face]) += total;
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
                    if (face.isBoundaryLoop())
                    {
                        continue;
                    }
                    total += a(fInds[face]) / 3.0;
                }
                out(vInds[vert]) += total;
            }
        }
    } // namespace Hs
} // namespace rsurfaces