#include "sobolev/hs.h"
#include "helpers.h"
#include "sobolev/all_constraints.h"
#include "spatial/convolution.h"
#include "spatial/convolution_kernel.h"
#include "sobolev/h1.h"
#include "block_cluster_tree.h"

#include "Eigen/Sparse"

namespace rsurfaces
{

    namespace Hs
    {
        inline Vector3 HatGradientOnTriangle(GCFace face, GCVertex vert, GeomPtr &geom)
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

        inline double HatAtTriangleCenter(GCFace face, GCVertex vert)
        {
            // If vert is adjacent to face, the value at the barycenter is 1/3
            for (GCVertex v : face.adjacentVertices())
            {
                if (v == vert)
                {
                    return 1. / 3.;
                }
            }
            // Otherwise, the value is 0
            return 0;
        }

        inline void AddTriangleGradientTerm(Eigen::MatrixXd &M, double s, GCFace f1, GCFace f2, GeomPtr &geom, VertexIndices &indices)
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

        inline void AddTriangleCenterTerm(Eigen::MatrixXd &M, double s, GCFace f1, GCFace f2, GeomPtr &geom, VertexIndices &indices)
        {
            std::vector<GCVertex> verts;
            GetVerticesWithoutDuplicates(f1, f2, verts);

            double area1 = geom->faceArea(f1);
            double area2 = geom->faceArea(f2);

            Vector3 mid1 = faceBarycenter(geom, f1);
            Vector3 mid2 = faceBarycenter(geom, f2);

            double dist_term = MetricDistanceTermFrac(s, mid1, mid2);

            for (GCVertex u : verts)
            {
                for (GCVertex v : verts)
                {
                    double u_hat_f1 = HatAtTriangleCenter(f1, u);
                    double u_hat_f2 = HatAtTriangleCenter(f2, u);
                    double v_hat_f1 = HatAtTriangleCenter(f1, v);
                    double v_hat_f2 = HatAtTriangleCenter(f2, v);

                    double numer = (u_hat_f1 - u_hat_f2) * (v_hat_f1 - v_hat_f2);
                    M(indices[u], indices[v]) += numer * dist_term * area1 * area2;
                }
            }
        }

        double get_s(double alpha, double beta)
        {
            return (beta - 2.0) / alpha;
        }

        void FillMatrixHigh(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom)
        {
            VertexIndices indices = mesh->getVertexIndices();
            for (GCFace f1 : mesh->faces())
            {
                for (GCFace f2 : mesh->faces())
                {
                    if (f1 == f2)
                        continue;
                    AddTriangleGradientTerm(M, s, f1, f2, geom, indices);
                }
            }
        }

        void FillMatrixFracOnly(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom)
        {
            VertexIndices indices = mesh->getVertexIndices();

            for (GCFace f1 : mesh->faces())
            {
                for (GCFace f2 : mesh->faces())
                {
                    if (f1 == f2)
                        continue;
                    AddTriangleCenterTerm(M, s, f1, f2, geom, indices);
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
            FillMatrixHigh(M_small, get_s(alpha, beta), mesh, geom);
            // Add single row in small block for barycenter
            Constraints::BarycenterConstraint bconstraint;
            Constraints::addEntriesToSymmetric(bconstraint, M_small, mesh, geom, nVerts);
            // Reduplicate entries 3x along diagonals; barycenter row gets tripled
            MatrixUtils::TripleMatrix(M_small, M);
            // Add rows for scaling to tripled block
            // Constraints::addScalingEntries(M, mesh, geom, 3 * nVerts + 3);

            // Flatten the gradient into a single column
            Eigen::VectorXd gradientCol;
            gradientCol.setZero(dims);

            MatrixUtils::MatrixIntoColumn(gradient, gradientCol);
            MatrixUtils::SolveDenseSystem(M, gradientCol, gradientCol);
            MatrixUtils::ColumnIntoMatrix(gradientCol, dest);
        }

        void ProjectViaSparse(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom, BVHNode6D *bvh)
        {
            // Reshape the gradient N x 3 matrix into a 3N-vector.
            Eigen::VectorXd gradientCol;
            gradientCol.setZero(3 * mesh->nVertices() + 3);
            MatrixUtils::MatrixIntoColumn(gradient, gradientCol);

            ProjectViaSparse(gradientCol, gradientCol, alpha, beta, mesh, geom, bvh);

            // Reshape it back into the output N x 3 matrix.
            MatrixUtils::ColumnIntoMatrix(gradientCol, dest);
        }

        void ProjectViaSparse(Eigen::VectorXd &gradientCol, Eigen::VectorXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom, BVHNode6D *bvh)
        {
            double s = Hs::get_s(alpha, beta);

            // Assemble the cotan Laplacian
            std::vector<Triplet> triplets, triplets3x;
            H1::getTriplets(triplets, mesh, geom);
            // Add a constraint row / col corresponding to barycenter weights
            Constraints::BarycenterConstraint bconstraint;
            Constraints::addTripletsToSymmetric(bconstraint, triplets, mesh, geom, mesh->nVertices());
            // Expand the matrix by 3x
            MatrixUtils::TripleTriplets(triplets, triplets3x);
            // Pre-factorize the cotan Laplacian
            Eigen::SparseMatrix<double> L(3 * mesh->nVertices() + 3, 3 * mesh->nVertices() + 3);
            L.setFromTriplets(triplets3x.begin(), triplets3x.end());
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> L_inv;
            L_inv.compute(L);


            // Multiply by L^{-1} once by solving Lx = b
            gradientCol = L_inv.solve(gradientCol);

            if (!bvh)
            {
                std::cout << "  * Assembling dense matrix to multiply" << std::endl;
                // Multiply by L^{2 - s}, a fractional Laplacian; this has order 4 - 2s
                Eigen::MatrixXd M, M3;
                M.setZero(mesh->nVertices(), mesh->nVertices());
                M3.setZero(3 * mesh->nVertices() + 3, 3 * mesh->nVertices() + 3);
                Hs::FillMatrixFracOnly(M, 4 - 2 * s, mesh, geom);
                MatrixUtils::TripleMatrix(M, M3);
                gradientCol = M3 * gradientCol;
            }

            else
            {
                std::cout << "  * Using block cluster tree to multiply" << std::endl;
                BlockClusterTree *bct = new BlockClusterTree(mesh, geom, bvh, 0.5, 4 - 2 * s);
                bct->MultiplyVector3(gradientCol, gradientCol);
                delete bct;
            }

            // Multiply by L^{-1} again by solving Lx = b
            dest = L_inv.solve(gradientCol);
        }

        void ProjectViaSchur(std::vector<ConstraintBase *> constraints, Eigen::MatrixXd &gradient,
                             Eigen::MatrixXd &dest, double alpha, double beta, MeshPtr &mesh, GeomPtr &geom, BVHNode6D *bvh)
        {
            size_t nVerts = mesh->nVertices();
            size_t nRows = 0;
            // Figure out how many rows the constraint block is
            for (ConstraintBase *c : constraints)
            {
                nRows += c->nRows();
            }

            if (nRows == 0) {
                std::cout << "No constraints provided to Schur complement method." << std::endl;
                throw 1;
            }

            Eigen::MatrixXd C;
            C.setZero(nRows, 3 * nVerts + 3);
            size_t curRow = 0;
            // Fill in the constraint block by getting the entries for each constraint
            // while incrementing the rows
            for (ConstraintBase *c : constraints)
            {
                c->addEntries(C, mesh, geom, curRow);
                curRow += c->nRows();
            }

            // https://en.wikipedia.org/wiki/Schur_complement
            // We want to compute (M/A) = D - C A^{-1} B.
            // In our case, D = 0, and B = C^T, so this is C A^{-1} C^T.
            // Unfortunately this means we have to apply A^{-1} once for each column of C^T,
            // which could get expensive if we have too many constraints.

            // First allocate some space for a single column
            Eigen::VectorXd curCol;
            curCol.setZero(3 * nVerts + 3);
            // And some space for A^{-1} C^T
            Eigen::MatrixXd A_inv_CT;
            A_inv_CT.setZero(3 * nVerts + 3, nRows);

            // For each column, copy it into curCol, and do the solve for A^{-1}
            for (size_t r = 0; r < nRows; r++)
            {
                // Copy the row of C into the column
                for (size_t i = 0; i < 3 * nVerts; i++)
                {
                    curCol(i) = C(r, i);
                }
                ProjectViaSparse(curCol, curCol, alpha, beta, mesh, geom, bvh);
                // Copy the column into the column of A^{-1} C^T
                for (size_t i = 0; i < 3 * nVerts + 3; i++)
                {
                    A_inv_CT(i, r) = curCol(i);
                }
            }

            // Now we've multiplied A^{-1} C^T, so just multiply this with C and negate it
            Eigen::MatrixXd M_A = -C * A_inv_CT;

            // We can do the actual inversion of the "saddle matrix" now:
            // the block of M^{-1} we want is A^{-1} + A^{-1} B (M/A)^{-1} C A^{-1}

            // Reuse curCol to store A^{-1} x
            curCol.setZero();
            MatrixUtils::MatrixIntoColumn(gradient, curCol);
            ProjectViaSparse(curCol, curCol, alpha, beta, mesh, geom, bvh);

            // Now we compute the correction.
            // Again we already have A^{-1} once, so no need to recompute it
            Eigen::VectorXd C_Ai_x = C * curCol;
            Eigen::VectorXd MAi_C_Ai_x;
            MAi_C_Ai_x.setZero(C_Ai_x.rows());
            MatrixUtils::SolveDenseSystem(M_A, C_Ai_x, MAi_C_Ai_x);
            Eigen::VectorXd B_MAi_C_Ai_x = C.transpose() * MAi_C_Ai_x;
            // Apply A^{-1} from scratch one more time
            ProjectViaSparse(B_MAi_C_Ai_x, B_MAi_C_Ai_x, alpha, beta, mesh, geom, bvh);

            curCol = curCol + B_MAi_C_Ai_x;
            MatrixUtils::ColumnIntoMatrix(curCol, dest);
        }
    } // namespace Hs

} // namespace rsurfaces