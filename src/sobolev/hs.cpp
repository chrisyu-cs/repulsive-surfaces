#include "sobolev/hs.h"
#include "helpers.h"
#include "spatial/convolution.h"
#include "spatial/convolution_kernel.h"

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

        HsMetric::HsMetric(SurfaceEnergy *energy_)
        {
            initFromEnergy(energy_);
            energy = energy_;
            // If no constraints are supplied, then we add our own barycenter
            // constraint as a default
            simpleConstraints.push_back(new Constraints::BarycenterConstraint3X(mesh, geom));
            // Remember to delete our constraint if we made our own
            usedDefaultConstraint = true;
            schurComplementComputed = false;
            allowBarycenterShift = false;
            precomputeSizes();
        }

        HsMetric::HsMetric(SurfaceEnergy *energy_, std::vector<SimpleProjectorConstraint *> &spcs, std::vector<ConstraintPack> &schurs)
            : simpleConstraints(spcs), newtonConstraints(schurs)
        {
            initFromEnergy(energy_);
            energy = energy_;
            usedDefaultConstraint = false;
            schurComplementComputed = false;
            precomputeSizes();
        }

        void HsMetric::initFromEnergy(SurfaceEnergy *energy_)
        {
            mesh = energy_->GetMesh();
            geom = energy_->GetGeom();
            bvh = energy_->GetBVH();
            bh_theta = energy_->GetTheta();
            bct = 0;
        }

        void HsMetric::precomputeSizes()
        {
            simpleRows = 0;
            for (Constraints::SimpleProjectorConstraint *spc : simpleConstraints)
            {
                simpleRows += spc->nRows();
                if (dynamic_cast<Constraints::BarycenterConstraint3X *>(spc))
                {
                    std::cout << "TODO: swap barycenter for component version?" << std::endl;
                }
            }

            newtonRows = 0;
            for (const ConstraintPack &schur : newtonConstraints)
            {
                newtonRows += schur.constraint->nRows();
            }
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

        void HsMetric::addSimpleConstraintEntries(Eigen::MatrixXd &M) const
        {
            size_t curRow = 3 * mesh->nVertices();
            for (SimpleProjectorConstraint *spc : simpleConstraints)
            {
                Constraints::addEntriesToSymmetric(*spc, M, mesh, geom, curRow);
                curRow += spc->nRows();
            }
        }

        void HsMetric::addSimpleConstraintTriplets(std::vector<Triplet> &triplets) const
        {
            size_t curRow = 3 * mesh->nVertices();
            for (SimpleProjectorConstraint *spc : simpleConstraints)
            {
                Constraints::addTripletsToSymmetric(*spc, triplets, mesh, geom, curRow);
                curRow += spc->nRows();
            }
        }

        Eigen::MatrixXd HsMetric::GetHsMatrixConstrained() const
        {
            Eigen::MatrixXd M_small, M;
            int nVerts = mesh->nVertices();
            M_small.setZero(nVerts, nVerts);
            int dims = getNumRows();

            VertexIndices inds = mesh->getVertexIndices();

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

            if (usesOnlyBarycenter())
            {
                Eigen::VectorXd masses;
                masses.setZero(topLeftNumRows());

                for (GCVertex v : mesh->vertices())
                {
                    size_t baserow = 3 * inds[v];
                    double wt = geom->vertexDualAreas[v];
                    masses(baserow) = wt;
                    masses(baserow + 1) = wt;
                    masses(baserow + 2) = wt;
                }

                Eigen::SparseMatrix<double> B = BarycenterQ() * masses.asDiagonal();

                double totalA = totalArea(geom, mesh);
                double gamma = pow(totalA, -getExpS());

                M.block(0, 0, B.cols(), B.cols()) += gamma * (B.transpose() * B);
            }

            for (const ConstraintPack &pack : newtonConstraints)
            {
                Constraints::addEntriesToSymmetric(*pack.constraint, M, mesh, geom, curRow);
                curRow += pack.constraint->nRows();
            }

            return M;
        }

        Eigen::SparseMatrix<double> HsMetric::GetConstraintBlock(bool includeNewton) const
        {
            std::vector<Triplet> triplets;
            size_t curRow = 0;

            for (SimpleProjectorConstraint *cons : simpleConstraints)
            {
                cons->addTriplets(triplets, mesh, geom, curRow);
                curRow += cons->nRows();
            }

            if (includeNewton)
            {
                for (const ConstraintPack &pack : newtonConstraints)
                {
                    pack.constraint->addTriplets(triplets, mesh, geom, curRow);
                    curRow += pack.constraint->nRows();
                }
            }

            Eigen::SparseMatrix<double> C(curRow, 3 * mesh->nVertices());
            C.setFromTriplets(triplets.begin(), triplets.end());

            return C;
        }

        void HsMetric::ProjectGradientExact(const Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, Eigen::PartialPivLU<Eigen::MatrixXd> &solver) const
        {
            Eigen::MatrixXd M = GetHsMatrixConstrained();

            // Flatten the gradient into a single column
            Eigen::VectorXd gradientCol;
            gradientCol.setZero(M.rows());

            MatrixUtils::MatrixIntoColumn(gradient, gradientCol);
            MatrixUtils::SolveDenseSystem(M, gradientCol, gradientCol);
            MatrixUtils::ColumnIntoMatrix(gradientCol, dest);
        }

        void HsMetric::ProjectSparseMat(const Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, double epsilon) const
        {
            // Reshape the gradient N x 3 matrix into a 3N-vector.
            Eigen::VectorXd gradientCol, gradientColProj;
            gradientCol.setZero(topLeftNumRows());
            MatrixUtils::MatrixIntoColumn(gradient, gradientCol);

            ProjectSparse(gradientCol, gradientCol, epsilon);

            // Reshape it back into the output N x 3 matrix.
            MatrixUtils::ColumnIntoMatrix(gradientCol, dest);
        }

        void HsMetric::shiftBarycenterConstraint(Vector3 shift)
        {
            for (SimpleProjectorConstraint *spc : simpleConstraints)
            {
                Constraints::BarycenterConstraint3X *bcenter = dynamic_cast<Constraints::BarycenterConstraint3X *>(spc);
                if (bcenter)
                {
                    bcenter->addShiftToCenter(shift);
                }
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

        void HsMetric::initLaplacianBarycenterMode() const
        {
            std::cout << "Initializing Laplacian for barycenter-only mode" << std::endl;
            size_t nRows = topLeftNumRows();
            Constraints::BarycenterComponentsConstraint *bCons = dynamic_cast<Constraints::BarycenterComponentsConstraint *>(simpleConstraints[0]);
            VertexIndices inds = mesh->getVertexIndices();

            // Assemble the cotan Laplacian
            std::vector<Triplet> laplaceTriplets, laplaceTriplets3x;
            H1::getTriplets(laplaceTriplets, mesh, geom, 1e-10);

            // Add the special barycenter triplets
            std::vector<Triplet> qTripletList(mesh->nVertices());
            bCons->getInnerLaplacianTriplets1X(qTripletList, mesh, geom, inds, 0);

            std::vector<Triplet> qTriplets3x;

            // The triplets now need to go into two places: the Laplacian itself
            // (weighted by mass, starting from row 3V),
            // and the matrix Q (unweighted, starting from row 0)
            for (GCVertex v : mesh->vertices())
            {
                // For Q we don't need to do anything
                Triplet q_v = qTripletList[inds[v]];
                // B goes in the Laplacian, so we offset the rows by V
                size_t lr = mesh->nVertices() + q_v.row();
                size_t lc = q_v.col();
                // B = QM, where M is just the masses; this is equivalent
                // to multiplying each row by the vertex mass
                double lval = q_v.value() * geom->vertexDualAreas[v];
                // Add symmetrically
                laplaceTriplets.push_back(Triplet(lr, lc, lval));
                laplaceTriplets.push_back(Triplet(lc, lr, lval));
            }

            // Expand the matrix by 3x
            MatrixUtils::TripleTriplets(laplaceTriplets, laplaceTriplets3x);
            MatrixUtils::TripleTriplets(qTripletList, qTriplets3x);

            // Pre-factorize the cotan Laplacian
            Eigen::SparseMatrix<double> L(nRows, nRows);
            L.setFromTriplets(laplaceTriplets3x.begin(), laplaceTriplets3x.end());
            std::cout << "Created Laplacian (" << L.rows() << " x " << L.cols() << ")" << std::endl;
            factorizedLaplacian.Compute(L);

            // No need to factorize Q, but we will apply it later
            barycenterQ.resize(bCons->nRows(), nRows);
            barycenterQ.setFromTriplets(qTriplets3x.begin(), qTriplets3x.end());
            std::cout << "Created Q matrix (" << barycenterQ.rows() << " x " << barycenterQ.cols() << ")" << std::endl;

            // Compute the weight for the Q^T Q term later
            double meshArea = totalArea(geom, mesh);
            double s = get_s(energy->GetExponents());
            // Exponent is 2s / n, but since n = 2, this is just s
            QTQ_weight = pow(meshArea, s);
            std::cout << "Computed weight = " << QTQ_weight << std::endl;
        }
    } // namespace Hs
} // namespace rsurfaces