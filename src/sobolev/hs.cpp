#include "sobolev/hs.h"
#include "helpers.h"
#include "sobolev/all_constraints.h"
#include "spatial/convolution.h"
#include "spatial/convolution_kernel.h"
#include "sobolev/h1.h"

namespace rsurfaces
{
    namespace Hs
    {
        using Constraints::SimpleProjectorConstraint;

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

        HsMetric::HsMetric(SurfaceEnergy *energy_)
        {
            initFromEnergy(energy_);
            // If no constraints are supplied, then we add our own barycenter
            // constraint as a default
            simpleConstraints.push_back(new Constraints::BarycenterConstraint3X(mesh, geom));
            // Remember to delete our constraint if we made our own
            usedDefaultConstraint = true;
        }

        HsMetric::HsMetric(SurfaceEnergy *energy_, std::vector<SimpleProjectorConstraint *> &spcs)
        {
            initFromEnergy(energy_);
            usedDefaultConstraint = false;

            if (spcs.size() == 0)
            {
                std::cerr << "ERROR: Need at least one constraint to initialize HsMetric." << std::endl;
                std::exit(1);
            }

            for (SimpleProjectorConstraint *spc : spcs)
            {
                // Push pointers to the existing constraints, which should
                // exist in SurfaceFlow (and not be allocated here)
                simpleConstraints.push_back(spc);
            }
        }

        void HsMetric::initFromEnergy(SurfaceEnergy *energy_)
        {
            Vector2 ab = energy_->GetExponents();
            mesh = energy_->GetMesh();
            geom = energy_->GetGeom();
            order_s = get_s(ab.x, ab.y);
            bvh = energy_->GetBVH();
            bh_theta = energy_->GetTheta();
            bct = 0;
        }

        HsMetric::~HsMetric()
        {
            if (bct)
            {
                delete bct;
            }
            if (usedDefaultConstraint)
            {
                for (size_t i = 0; i < simpleConstraints.size(); i++)
                {
                    delete simpleConstraints[i];
                }
            }
        }

        void HsMetric::FillMatrixHigh(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom)
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

        void HsMetric::FillMatrixFracOnly(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom)
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

        void HsMetric::FillMatrixVertsFirst(Eigen::MatrixXd &M, double s, MeshPtr &mesh, GeomPtr &geom)
        {
            VertexIndices indices = mesh->getVertexIndices();
            std::vector<GCFace> faces;

            for (GCVertex u : mesh->vertices())
            {
                for (GCVertex v : mesh->vertices())
                {
                    faces.clear();
                    GetFacesWithoutDuplicates(u, v, faces);
                    double A_uv = 0;

                    for (GCFace f1 : faces)
                    {
                        for (GCFace f2 : faces)
                        {
                            if (f1 == f2)
                            {
                                continue;
                            }
                            double u_hat_f1 = HatAtTriangleCenter(f1, u);
                            double u_hat_f2 = HatAtTriangleCenter(f2, u);
                            double v_hat_f1 = HatAtTriangleCenter(f1, v);
                            double v_hat_f2 = HatAtTriangleCenter(f2, v);
                            double dist_term = MetricDistanceTermFrac(s, faceBarycenter(geom, f1), faceBarycenter(geom, f2));

                            double area1 = geom->faceAreas[f1];
                            double area2 = geom->faceAreas[f2];

                            double numer = (u_hat_f1 - u_hat_f2) * (v_hat_f1 - v_hat_f2);
                            A_uv += numer * dist_term * area1 * area2;
                        }
                    }
                    M(indices[u], indices[v]) = A_uv;
                }
            }
        }

        size_t HsMetric::topLeftNumRows()
        {
            size_t nConstraints = 0;
            for (SimpleProjectorConstraint *spc : simpleConstraints)
            {
                nConstraints += spc->nRows();
            }
            return 3 * mesh->nVertices() + nConstraints;
        }

        void HsMetric::addSimpleConstraintEntries(Eigen::MatrixXd &M)
        {
            size_t curRow = 3 * mesh->nVertices();
            for (SimpleProjectorConstraint *spc : simpleConstraints)
            {
                Constraints::addEntriesToSymmetric(*spc, M, mesh, geom, curRow);
                curRow += spc->nRows();
            }
        }

        void HsMetric::addSimpleConstraintTriplets(std::vector<Triplet> &triplets)
        {
            size_t curRow = 3 * mesh->nVertices();
            for (SimpleProjectorConstraint *spc : simpleConstraints)
            {
                Constraints::addTripletsToSymmetric(*spc, triplets, mesh, geom, curRow);
                curRow += spc->nRows();
            }
        }

        void HsMetric::ProjectGradient(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest)
        {
            // Assemble the metric matrix
            Eigen::MatrixXd M_small, M;
            int nVerts = mesh->nVertices();
            M_small.setZero(nVerts + 1, nVerts + 1);
            int dims = topLeftNumRows();

            M.setZero(dims, dims);
            FillMatrixHigh(M_small, order_s, mesh, geom);
            // Reduplicate entries 3x along diagonals; barycenter row gets tripled
            MatrixUtils::TripleMatrix(M_small, M);
            // Add rows in tripled block for barycenter constraint (and potentially others)
            addSimpleConstraintEntries(M);

            // Flatten the gradient into a single column
            Eigen::VectorXd gradientCol;
            gradientCol.setZero(dims);

            MatrixUtils::MatrixIntoColumn(gradient, gradientCol);
            MatrixUtils::SolveDenseSystem(M, gradientCol, gradientCol);
            MatrixUtils::ColumnIntoMatrix(gradientCol, dest);
        }

        void HsMetric::ProjectViaSparseMat(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest)
        {
            // Reshape the gradient N x 3 matrix into a 3N-vector.
            Eigen::VectorXd gradientCol, gradientColProj;
            gradientCol.setZero(topLeftNumRows());
            MatrixUtils::MatrixIntoColumn(gradient, gradientCol);

            ProjectViaSparse(gradientCol, gradientCol);

            // Reshape it back into the output N x 3 matrix.
            MatrixUtils::ColumnIntoMatrix(gradientCol, dest);
        }

        void HsMetric::ProjectViaSparse(Eigen::VectorXd &gradientCol, Eigen::VectorXd &dest)
        {
            size_t nRows = topLeftNumRows();

            if (!factorizedLaplacian.initialized)
            {
                // Assemble the cotan Laplacian
                std::vector<Triplet> triplets, triplets3x;
                H1::getTriplets(triplets, mesh, geom);
                // Expand the matrix by 3x
                MatrixUtils::TripleTriplets(triplets, triplets3x);

                // Add constraint rows / cols for "simple" constraints included in Laplacian
                addSimpleConstraintTriplets(triplets3x);
                // Pre-factorize the cotan Laplacian
                Eigen::SparseMatrix<double> L(nRows, nRows);
                L.setFromTriplets(triplets3x.begin(), triplets3x.end());
                factorizedLaplacian.Compute(L);
            }

            // Multiply by L^{-1} once by solving Lx = b
            gradientCol = factorizedLaplacian.Solve(gradientCol);

            if (!bvh)
            {
                std::cout << "  * Assembling dense matrix to multiply" << std::endl;
                // Multiply by L^{2 - s}, a fractional Laplacian; this has order 4 - 2s
                Eigen::MatrixXd M, M3;
                M.setZero(mesh->nVertices(), mesh->nVertices());
                M3.setZero(nRows, nRows);
                FillMatrixFracOnly(M, 4 - 2 * order_s, mesh, geom);
                MatrixUtils::TripleMatrix(M, M3);
                gradientCol = M3 * gradientCol;
            }

            else
            {
                if (!bct)
                {
                    long bctConstructStart = currentTimeMilliseconds();
                    bct = new BlockClusterTree(mesh, geom, bvh, bh_theta, 4 - 2 * order_s);
                    long bctConstructEnd = currentTimeMilliseconds();
                    std::cout << "    * BCT construction = " << (bctConstructEnd - bctConstructStart) << " ms" << std::endl;
                }
                long bctMultStart = currentTimeMilliseconds();
                bct->MultiplyVector3(gradientCol, gradientCol);
                long bctMultEnd = currentTimeMilliseconds();

                std::cout << "    * BCT multiply = " << (bctMultEnd - bctMultStart) << " ms" << std::endl;
            }

            // Re-zero out Lagrange multipliers, since the first solve
            // will have left some junk in them
            for (size_t i = 3 * mesh->nVertices(); i < nRows; i++)
            {
                gradientCol(i) = 0;
            }

            // Multiply by L^{-1} again by solving Lx = b
            dest = factorizedLaplacian.Solve(gradientCol);
        }

        void HsMetric::GetSchurComplement(std::vector<ConstraintPack> constraints, SchurComplement &dest)
        {
            size_t nVerts = mesh->nVertices();
            size_t compNRows = 0;
            size_t bigNRows = topLeftNumRows();

            // Figure out how many rows the constraint block is
            for (ConstraintPack &c : constraints)
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
            for (ConstraintPack &c : constraints)
            {
                c.constraint->addEntries(dest.C, mesh, geom, curRow);
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
                ProjectViaSparse(curCol, curCol);
                // Copy the column into the column of A^{-1} C^T
                for (size_t i = 0; i < bigNRows; i++)
                {
                    A_inv_CT(i, r) = curCol(i);
                }
            }

            // Now we've multiplied A^{-1} C^T, so just multiply this with C and negate it
            dest.M_A = -dest.C * A_inv_CT;
        }

        void HsMetric::ProjectViaSchur(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, SchurComplement &comp)
        {
            size_t nVerts = mesh->nVertices();
            // Invert the "saddle matrix" now:
            // the block of M^{-1} we want is A^{-1} + A^{-1} C^T (M/A)^{-1} C A^{-1}

            // Reuse curCol to store A^{-1} x
            // First allocate some space for a single column
            Eigen::VectorXd curCol;
            curCol.setZero(comp.C.cols());
            MatrixUtils::MatrixIntoColumn(gradient, curCol);
            ProjectViaSparse(curCol, curCol);

            // Now we compute the correction.
            // Again we already have A^{-1} once, so no need to recompute it
            Eigen::VectorXd C_Ai_x = comp.C * curCol;
            Eigen::VectorXd MAi_C_Ai_x;
            MAi_C_Ai_x.setZero(C_Ai_x.rows());
            MatrixUtils::SolveDenseSystem(comp.M_A, C_Ai_x, MAi_C_Ai_x);
            Eigen::VectorXd B_MAi_C_Ai_x = comp.C.transpose() * MAi_C_Ai_x;
            // Apply A^{-1} from scratch one more time
            ProjectViaSparse(B_MAi_C_Ai_x, B_MAi_C_Ai_x);

            curCol = curCol + B_MAi_C_Ai_x;
            MatrixUtils::ColumnIntoMatrix(curCol, dest);
        }

        void HsMetric::ProjectSchurConstraints(std::vector<ConstraintPack> &constraints, SchurComplement &comp)
        {
            size_t nRows = comp.M_A.rows();
            int nIters = 0;
            Eigen::VectorXd vals(nRows);

            while (nIters < 1)
            {
                vals.setZero();
                int curRow = 0;
                // Fill right-hand side with error values
                for (ConstraintPack &c : constraints)
                {
                    c.constraint->addErrorValues(vals, mesh, geom, curRow);
                    curRow += c.constraint->nRows();
                }

                double constraintError = vals.lpNorm<Eigen::Infinity>();
                std::cout << "Constraint error after " << nIters << " iterations = " << constraintError << std::endl;
                if (nIters > 0 && constraintError < 1e-3)
                {
                    break;
                }

                nIters++;

                // In this case we want the block of the inverse that multiplies the bottom block
                // -A^{-1} B (M/A)^{-1}, where B = C^T
                // Apply (M/A) inverse first
                MatrixUtils::SolveDenseSystem(comp.M_A, vals, vals);
                // Apply C^T
                Eigen::VectorXd correction = comp.C.transpose() * vals;
                // Apply A^{-1}
                ProjectViaSparse(correction, correction);

                // Apply the correction to the vertex positions
                VertexIndices verts = mesh->getVertexIndices();
                size_t nVerts = mesh->nVertices();
                for (GCVertex v : mesh->vertices())
                {
                    int base = 3 * verts[v];
                    Vector3 vertCorr{correction(base), correction(base + 1), correction(base + 2)};
                    geom->inputVertexPositions[v] += vertCorr;
                }

                vals.setZero();
                curRow = 0;
                // Fill right-hand side with error values
                for (ConstraintPack &c : constraints)
                {
                    c.constraint->addErrorValues(vals, mesh, geom, curRow);
                    curRow += c.constraint->nRows();
                }
                double corrError = vals.lpNorm<Eigen::Infinity>();
                std::cout << "Corrected error " << constraintError << " -> " << corrError << std::endl;
            }
        }

        void HsMetric::ProjectSimpleConstraints()
        {
            for (SimpleProjectorConstraint *spc : simpleConstraints)
            {
                spc->ProjectConstraint(mesh, geom);
            }
        }

        void HsMetric::ProjectSimpleConstraintsWithSaddle()
        {
            Eigen::VectorXd vals(factorizedLaplacian.nRows);
            vals.setZero();
            int baseRow = 3 * mesh->nVertices();
            int currRow = baseRow;
            // Fill the right-hand side with error values
            for (SimpleProjectorConstraint *spc : simpleConstraints)
            {
                spc->addErrorValues(vals, mesh, geom, currRow);
                currRow += spc->nRows();
            }
            // Solve for the correction
            Eigen::VectorXd corr = factorizedLaplacian.Solve(vals);

            // Apply the correction
            VertexIndices verts = mesh->getVertexIndices();
            for (GCVertex v : mesh->vertices())
            {
                size_t base = 3 * verts[v];
                Vector3 vertCorr{corr(base), corr(base + 1), corr(base + 2)};
                geom->inputVertexPositions[v] -= vertCorr;
            }
        }
    } // namespace Hs
} // namespace rsurfaces