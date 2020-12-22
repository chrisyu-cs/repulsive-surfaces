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

        inline Vector3 HatGradientOnTriangle(const GCFace face, const GCVertex vert, const GeomPtr &geom)
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

            // If vert isn't adjacent, value is 0
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

        inline double HatAtTriangleCenter(const GCFace face, const GCVertex vert)
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

        inline void AddTriangleGradientTerm(Eigen::MatrixXd &M, double s, GCFace f1, GCFace f2, const GeomPtr &geom, VertexIndices &indices)
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

        inline void AddTriangleCenterTerm(Eigen::MatrixXd &M, double s, GCFace f1, GCFace f2, const GeomPtr &geom, VertexIndices &indices, bool lowOrder)
        {
            std::vector<GCVertex> verts;
            GetVerticesWithoutDuplicates(f1, f2, verts);

            double area1 = geom->faceArea(f1);
            double area2 = geom->faceArea(f2);

            Vector3 mid1 = faceBarycenter(geom, f1);
            Vector3 mid2 = faceBarycenter(geom, f2);

            double dist_term = 0;

            if (lowOrder)
            {
                Vector3 n1 = faceNormal(geom, f1);
                Vector3 n2 = faceNormal(geom, f2);
                dist_term = MetricDistanceTermLowPure(s, mid1, mid2, n1, n2);
            }
            else
            {
                dist_term = MetricDistanceTermFrac(s, mid1, mid2);
            }

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
            energy = energy_;
            // If no constraints are supplied, then we add our own barycenter
            // constraint as a default
            simpleConstraints.push_back(new Constraints::BarycenterConstraint3X(mesh, geom));
            // Remember to delete our constraint if we made our own
            usedDefaultConstraint = true;
        }

        HsMetric::HsMetric(SurfaceEnergy *energy_, std::vector<SimpleProjectorConstraint *> &spcs)
        {
            initFromEnergy(energy_);
            energy = energy_;
            usedDefaultConstraint = false;

            for (SimpleProjectorConstraint *spc : spcs)
            {
                // Push pointers to the existing constraints, which should
                // exist in SurfaceFlow (and not be allocated here)
                simpleConstraints.push_back(spc);
            }
        }

        void HsMetric::initFromEnergy(SurfaceEnergy *energy_)
        {
            mesh = energy_->GetMesh();
            geom = energy_->GetGeom();
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

        void HsMetric::FillMatrixHigh(Eigen::MatrixXd &M, double s, const MeshPtr &mesh, const GeomPtr &geom) const
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

        void HsMetric::FillMatrixLow(Eigen::MatrixXd &M, double s, const MeshPtr &mesh, const GeomPtr &geom) const
        {
            VertexIndices indices = mesh->getVertexIndices();

            for (GCFace f1 : mesh->faces())
            {
                for (GCFace f2 : mesh->faces())
                {
                    if (f1 == f2)
                        continue;
                    AddTriangleCenterTerm(M, s, f1, f2, geom, indices, true);
                }
            }
        }

        void HsMetric::FillMatrixFracOnly(Eigen::MatrixXd &M, double s, const MeshPtr &mesh, const GeomPtr &geom) const
        {
            VertexIndices indices = mesh->getVertexIndices();

            for (GCFace f1 : mesh->faces())
            {
                for (GCFace f2 : mesh->faces())
                {
                    if (f1 == f2)
                        continue;
                    AddTriangleCenterTerm(M, s, f1, f2, geom, indices, false);
                }
            }
        }

        size_t HsMetric::topLeftNumRows() const
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

        Eigen::MatrixXd HsMetric::GetHsMatrixConstrained(std::vector<ConstraintPack> &schurConstraints) const
        {
            Eigen::MatrixXd M_small, M;
            int nVerts = mesh->nVertices();
            M_small.setZero(nVerts, nVerts);
            int dims = getNumRows(schurConstraints);

            M.setZero(dims, dims);
            double s = getHsOrder();
            FillMatrixHigh(M_small, s, mesh, geom);
            FillMatrixLow(M_small, s, mesh, geom);
            // Reduplicate entries 3x along diagonals; barycenter row gets tripled
            MatrixUtils::TripleMatrix(M_small, M);

            size_t curRow = 3 * nVerts;

            for (SimpleProjectorConstraint *cons : simpleConstraints)
            {
                Constraints::addEntriesToSymmetric(*cons, M, mesh, geom, curRow);
                curRow += cons->nRows();
            }

            for (ConstraintPack &pack : schurConstraints)
            {
                Constraints::addEntriesToSymmetric(*pack.constraint, M, mesh, geom, curRow);
                curRow += pack.constraint->nRows();
            }

            return M;
        }

        Eigen::SparseMatrix<double> HsMetric::GetConstraintBlock(std::vector<ConstraintPack> &schurConstraints) const
        {
            std::vector<Triplet> triplets;
            size_t curRow = 0;

            for (SimpleProjectorConstraint *cons : simpleConstraints)
            {
                cons->addTriplets(triplets, mesh, geom, curRow);
                curRow += cons->nRows();
            }

            for (ConstraintPack &pack : schurConstraints)
            {
                pack.constraint->addTriplets(triplets, mesh, geom, curRow);
                curRow += pack.constraint->nRows();
            }

            Eigen::SparseMatrix<double> C(curRow, 3 * mesh->nVertices());
            C.setFromTriplets(triplets.begin(), triplets.end());

            return C;
        }


        void HsMetric::ProjectGradientExact(const Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, std::vector<ConstraintPack> &schurConstraints) const
        {
            Eigen::MatrixXd M = GetHsMatrixConstrained(schurConstraints);

            // Flatten the gradient into a single column
            Eigen::VectorXd gradientCol;
            gradientCol.setZero(M.rows());

            MatrixUtils::MatrixIntoColumn(gradient, gradientCol);
            MatrixUtils::SolveDenseSystem(M, gradientCol, gradientCol);
            MatrixUtils::ColumnIntoMatrix(gradientCol, dest);
        }

        void HsMetric::ProjectSparseMat(const Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest)
        {
            // Reshape the gradient N x 3 matrix into a 3N-vector.
            Eigen::VectorXd gradientCol, gradientColProj;
            gradientCol.setZero(topLeftNumRows());
            MatrixUtils::MatrixIntoColumn(gradient, gradientCol);

            ProjectSparse(gradientCol, gradientCol);

            // Reshape it back into the output N x 3 matrix.
            MatrixUtils::ColumnIntoMatrix(gradientCol, dest);
        }

        void HsMetric::ProjectSparse(const Eigen::VectorXd &gradientCol, Eigen::VectorXd &dest)
        {
            size_t nRows = topLeftNumRows();

            double epsilon = 1e-6;
            if (simpleConstraints.size() > 0)
            {
                epsilon = 1e-10;
            }

            if (!factorizedLaplacian.initialized)
            {
                // Assemble the cotan Laplacian
                std::vector<Triplet> triplets, triplets3x;
                H1::getTriplets(triplets, mesh, geom, epsilon);
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
            Eigen::VectorXd mid = factorizedLaplacian.Solve(gradientCol);

            if (!bvh)
            {
                throw std::runtime_error("Must have a BVH to use sparse approximation");
            }

            else
            {
                if (!bct)
                {
                    bct = new BlockClusterTree(mesh, geom, bvh, bh_theta, 4 - 2 * getHsOrder());
                }
                bct->MultiplyVector3(mid, mid, BCTKernelType::FractionalOnly);
            }

            // Re-zero out Lagrange multipliers, since the first solve
            // will have left some junk in them
            for (size_t i = 3 * mesh->nVertices(); i < nRows; i++)
            {
                mid(i) = 0;
            }

            // Multiply by L^{-1} again by solving Lx = b
            dest = factorizedLaplacian.Solve(mid);
        }

        void HsMetric::ProjectSparseWithR1Update(const Eigen::VectorXd &DE, Eigen::VectorXd &dest)
        {
            // We want to add a rank-1 update for DE * DE^T
            // First just compute A^{-1} x
            Eigen::VectorXd Ainv_x;
            Ainv_x.setZero(DE.rows());
            ProjectSparse(DE, Ainv_x);

            double currE = 1; //energy->Value();

            // Now we want to compute the numerator A^{-1} x x^T A^{-1} x
            // Right now "dest" already holds A^{-1} x, and "DE" holds x
            // First compute the scalar (x^T * A^{-1} * x)
            double xT_A_x = (DE.transpose() / currE) * Ainv_x;
            // Inner part of is x * (x^T * A^{-1} * x)
            Eigen::VectorXd numer = ((DE / currE) * xT_A_x);
            // Multiply by A^{-1} to get the whole thing
            ProjectSparse(numer, numer);

            // Denominator is (1 + x^T A^{-1} x)
            double denom = (1 + xT_A_x / currE);

            std::cout << "Ainv norm = " << Ainv_x.norm() << std::endl;
            std::cout << "Update norm = " << (numer / denom).norm() << std::endl;

            dest = Ainv_x - (numer / denom);
        }

        void HsMetric::ProjectSparseWithR1UpdateMat(const Eigen::MatrixXd &DE, Eigen::MatrixXd &dest)
        {
            // We want to add a rank-1 update for DE * DE^T
            // First just compute A^{-1} x
            Eigen::MatrixXd Ainv_x;
            Ainv_x.setZero(DE.rows(), DE.cols());
            ProjectSparseMat(DE, Ainv_x);

            // Now we want to compute the numerator A^{-1} x x^T A^{-1} x
            // Right now "Ainv_x" already holds A^{-1} x, and "DE" holds x
            // First compute the scalar (x^T * A^{-1} * x)
            double xT_A_x = (DE.transpose() * Ainv_x).trace();
            std::cout << "xT_A_x = " << xT_A_x << std::endl;
            // Inner part of is x * (x^T * A^{-1} * x)
            Eigen::MatrixXd numer = (DE * xT_A_x);
            // Multiply by A^{-1} to get the whole thing
            ProjectSparseMat(numer, numer);

            // Denominator is (1 + x^T A^{-1} x)
            double denom = (1 + xT_A_x);

            std::cout << "R1 update: " << numer.norm() << " / " << denom << std::endl;

            std::cout << "Ainv norm = " << Ainv_x.norm() << std::endl;
            std::cout << "Update norm = " << (numer / denom).norm() << std::endl;
            dest = Ainv_x - (numer / denom);
            std::cout << "Dest norm = " << dest.norm() << std::endl;
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
            if (simpleConstraints.size() == 0)
            {
                return;
            }

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