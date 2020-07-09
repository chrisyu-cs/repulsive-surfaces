#include "sobolev/hs.h"
#include "helpers.h"
#include "sobolev/constraints.h"
#include "spatial/convolution.h"
#include "spatial/convolution_kernel.h"

namespace rsurfaces
{

    namespace Hs
    {
        Vector3 HatGradientOnTriangle(GCFace face, GCVertex vert, GeomPtr &geom)
        {
            // Find the half-edge starting at vert in face
            GCHalfedge he = face.halfedge();
            GCHalfedge start = he;
            bool found = false;
            do
            {
                if (he.vertex() == vert)
                {
                    found = true;
                    break;
                }
                he = he.next();
            } while (start != he);

            if (!found)
            {
                return Vector3{0, 0, 0};
            }

            GCVertex nextVert = he.next().vertex();
            GCVertex nextNextVert = he.next().next().vertex();

            Vector3 oppDir = geom->inputVertexPositions[nextNextVert] - geom->inputVertexPositions[nextVert];
            oppDir = oppDir.normalize();
            Vector3 altitude = geom->inputVertexPositions[vert] - geom->inputVertexPositions[nextVert];
            // Project so that the altitude becomes normal to the opposite edge
            altitude = altitude - dot(oppDir, altitude) * oppDir;
            double length = norm(altitude);

            // Resulting gradient is (1 / length) * (normalized direction)
            return altitude / (length * length);
        }

        inline double MetricDistanceTerm(double s, Vector3 v1, Vector3 v2)
        {
            double dist_term = 1.0 / pow(norm(v1 - v2), 2 * (s - 1) + 2);
            return dist_term;
        }

        void AddTriangleContribution(Eigen::MatrixXd &M, double s, GCFace f1, GCFace f2, GeomPtr &geom, VertexIndices &indices)
        {
            std::vector<GCVertex> verts;
            GetVerticesWithoutDuplicates(f1, f2, verts);

            double area1 = geom->faceArea(f1);
            double area2 = geom->faceArea(f2);

            Vector3 mid1 = faceBarycenter(geom, f1);
            Vector3 mid2 = faceBarycenter(geom, f2);

            double dist_term = MetricDistanceTerm(s, mid1, mid2);

            for (GCVertex u : verts)
            {
                for (GCVertex v : verts)
                {
                    Vector3 u_hat_f1 = HatGradientOnTriangle(f1, u, geom);
                    Vector3 u_hat_f2 = HatGradientOnTriangle(f2, u, geom);
                    Vector3 v_hat_f1 = HatGradientOnTriangle(f1, v, geom);
                    Vector3 v_hat_f2 = HatGradientOnTriangle(f2, v, geom);

                    double numer = dot(u_hat_f1 - u_hat_f2, v_hat_f1 - v_hat_f2);
                    M(indices[u], indices[v]) += numer * dist_term * area1 * area2;
                }
            }
        }

        double get_s(double alpha, double beta)
        {
            return (beta - 2.0) / alpha;
        }

        void FillMatrix(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom)
        {
            VertexIndices indices = mesh->getVertexIndices();
            for (GCFace f1 : mesh->faces())
            {
                for (GCFace f2 : mesh->faces())
                {
                    if (f1 == f2)
                        continue;
                    AddTriangleContribution(M, s, f1, f2, geom, indices);
                }
            }
        }

        void ProjectGradient(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom)
        {
            // Assemble the metric matrix
            Eigen::MatrixXd M_small, M;
            int nVerts = mesh->nVertices();
            M_small.setZero(nVerts + 1, nVerts + 1);
            int dims = 3 * nVerts + 3;
            M.setZero(dims, dims);
            FillMatrix(M_small, get_s(alpha, beta), mesh, geom);
            Constraints::addBarycenterEntries(M_small, mesh, geom, nVerts);
            // Reduplicate entries 3x along diagonals
            MatrixUtils::TripleMatrix(M_small, M);

            // Flatten the gradient into a single column
            Eigen::VectorXd gradientCol;
            gradientCol.setZero(3 * mesh->nVertices() + 3);
            MatrixUtils::MatrixIntoColumn(gradient, gradientCol);

            // Invert the metric, and write it into the destination
            MatrixUtils::SolveDenseSystem(M, gradientCol, gradientCol);
            MatrixUtils::ColumnIntoMatrix(gradientCol, dest);
        }

        void ProjectViaConvolution(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom)
        {
            double s = get_s(alpha, beta);
            RieszKernel ker((2. - s) / 2.);
            Eigen::MatrixXd temp;
            temp.setZero(dest.rows(), dest.cols());

            ConvolveExact(mesh, geom, ker, gradient, temp);
            ConvolveExact(mesh, geom, ker, temp, dest);
        }
    } // namespace Hs

} // namespace rsurfaces