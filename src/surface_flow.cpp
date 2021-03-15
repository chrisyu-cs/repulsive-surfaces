#include "surface_flow.h"
#include "fractional_laplacian.h"
#include "helpers.h"

#include "sobolev/h1.h"
#include "sobolev/h1_lbfgs.h"
#include "sobolev/l2_lbfgs.h"
#include "sobolev/bqn_lbfgs.h"
#include "sobolev/hs.h"
#include "sobolev/hs_schur.h"
#include "sobolev/hs_iterative.h"
#include "sobolev/constraints.h"
#include "spatial/convolution.h"

#include <Eigen/SparseCholesky>

namespace rsurfaces
{

    SurfaceFlow::SurfaceFlow(SurfaceEnergy *energy_)
    {
        energies.push_back(energy_);

        mesh = energy_->GetMesh();
        geom = energy_->GetGeom();
        stepCount = 0;

        origBarycenter = meshBarycenter(geom, mesh);
        std::cout << "Original barycenter = " << origBarycenter << std::endl;
        RecenterMesh();
        secretBarycenter = 0;
        obstacleEnergy = 0;

        verticesMutated = false;
        lbfgs = 0;
        bqn_B = 0;
    }

    void SurfaceFlow::AddAdditionalEnergy(SurfaceEnergy *extraEnergy)
    {
        energies.push_back(extraEnergy);
    }

    void SurfaceFlow::AddObstacleEnergy(TPObstacleBarnesHut0 *obsEnergy)
    {
        obstacleEnergy = obsEnergy;
        AddAdditionalEnergy(obstacleEnergy);
    }


    void SurfaceFlow::UpdateEnergies()
    {
        for (SurfaceEnergy *energy : energies)
        {
            energy->Update();
        }
    }

    inline double guessStepSize(double gProjNorm)
    {
        // double initGuess = (gProjNorm < 1) ? 1.0 / sqrt(gProjNorm) : 1.0 / gProjNorm;
        double initGuess = 1.0 / gProjNorm;
        initGuess *= 2;
        return initGuess;
    }

    void SurfaceFlow::StepL2Unconstrained()
    {
        stepCount++;
        UpdateEnergies();

        Eigen::MatrixXd l2diff;
        l2diff.setZero(mesh->nVertices(), 3);
        AssembleGradients(l2diff);

        double initGuess = guessStepSize(l2diff.norm());
        LineSearch search(mesh, geom, energies);
        search.BacktrackingLineSearch(l2diff, initGuess, 1);
    }

    void SurfaceFlow::StepL2Projected()
    {
        stepCount++;
        UpdateEnergies();

        Eigen::MatrixXd l2diff;
        l2diff.setZero(mesh->nVertices(), 3);
        AssembleGradients(l2diff);

        // Make a saddle matrix with just the identity in the corner
        std::vector<Triplet> triplets;
        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            size_t i3 = 3 * i;
            triplets.push_back(Triplet(i3, i3, 1));
            triplets.push_back(Triplet(i3 + 1, i3 + 1, 1));
            triplets.push_back(Triplet(i3 + 2, i3 + 2, 1));
        }

        size_t nConstraintRows = addConstraintTriplets(triplets, true);
        size_t dims = 3 * mesh->nVertices() + nConstraintRows;

        Eigen::SparseMatrix<double> A(dims, dims);
        A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::VectorXd l2col;
        l2col.setZero(dims);
        MatrixUtils::MatrixIntoColumn(l2diff, l2col);

        SparseFactorization factorizedA;
        factorizedA.Compute(A);

        l2col = factorizedA.Solve(l2col);
        MatrixUtils::ColumnIntoMatrix(l2col, l2diff);

        double initGuess = guessStepSize(l2diff.norm());
        LineSearch search(mesh, geom, energies);
        search.BacktrackingLineSearch(l2diff, initGuess, 1);
        
        // Constraint projection
        l2col.setZero();
        size_t curRow = 3 * mesh->nVertices();

        for (Constraints::SimpleProjectorConstraint* spc : simpleConstraints)
        {
            spc->addErrorValues(l2col, mesh, geom, curRow);
            curRow += spc->nRows();
        }

        for (ConstraintPack pack : schurConstraints)
        {
            pack.constraint->addErrorValues(l2col, mesh, geom, curRow);
            curRow += pack.constraint->nRows();
        }

        l2col = factorizedA.Solve(l2col);

        VertexIndices inds = mesh->getVertexIndices();
        for (GCVertex v : mesh->vertices())
        {
            size_t base = inds[v] * 3;
            Vector3 corr{l2col(base), l2col(base + 1), l2col(base + 2)};
            geom->inputVertexPositions[v] -= corr;
        }

        geom->refreshQuantities();
    }


    double SurfaceFlow::evaluateEnergy()
    {
        return GetEnergyValue(energies);
    }

    void SurfaceFlow::AssembleGradients(Eigen::MatrixXd &dest)
    {
        AddGradientsToMatrix(energies, dest);
    }

    std::unique_ptr<Hs::HsMetric> SurfaceFlow::GetHsMetric()
    {
        std::unique_ptr<Hs::HsMetric> hs(new Hs::HsMetric(energies, obstacleEnergy, simpleConstraints, schurConstraints));
        hs->disableNearField = disableNearField;
        return hs;
    }

    void SurfaceFlow::StepProjectedGradientExact()
    {
        long timeStart = currentTimeMilliseconds();
        stepCount++;
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        std::cout << "Using Hs projected gradient method..." << std::endl;
        UpdateEnergies();

        // Assemble sum of L2 differentials of all energies involved
        // (including tangent-point energy)
        Eigen::MatrixXd l2diff, gradientProj;
        l2diff.setZero(mesh->nVertices(), 3);
        gradientProj.setZero(mesh->nVertices(), 3);

        AssembleGradients(l2diff);
        double gNorm = l2diff.norm();

        std::unique_ptr<Hs::HsMetric> hs = GetHsMetric();

        Eigen::MatrixXd M = hs->GetHsMatrixConstrained();

        // Flatten the gradient into a single column
        Eigen::VectorXd gradientCol;
        gradientCol.setZero(M.rows());

        // Solve the dense system
        MatrixUtils::MatrixIntoColumn(l2diff, gradientCol);
        Eigen::PartialPivLU<Eigen::MatrixXd> solver = M.partialPivLu();
        gradientCol = solver.solve(gradientCol);
        MatrixUtils::ColumnIntoMatrix(gradientCol, gradientProj);

        VertexIndices inds = mesh->getVertexIndices();

        double gProjNorm = gradientProj.norm();
        // Measure dot product of search direction with original gradient direction
        double gradDot = (l2diff.transpose() * gradientProj).trace() / (gNorm * gProjNorm);

        // Guess a step size
        // double initGuess = prevStep * 1.25;
        double initGuess = guessStepSize(gProjNorm);

        std::cout << "  * Initial step size guess = " << initGuess << std::endl;

        // Take the step using line search
        LineSearch search(mesh, geom, energies);
        search.BacktrackingLineSearch(gradientProj, initGuess, gradDot);
        geom->refreshQuantities();

        // Reuse factorized matrix for constraint projection
        gradientCol.setZero();
        size_t curRow = 3 * mesh->nVertices();

        for (Constraints::SimpleProjectorConstraint *cons : simpleConstraints)
        {
            cons->addErrorValues(gradientCol, mesh, geom, curRow);
            curRow += cons->nRows();
        }

        for (const ConstraintPack &c : schurConstraints)
        {
            c.constraint->addErrorValues(gradientCol, mesh, geom, curRow);
            curRow += c.constraint->nRows();
        }

        gradientCol = solver.solve(gradientCol);

        VertexIndices verts = mesh->getVertexIndices();
        for (GCVertex v : mesh->vertices())
        {
            int base = 3 * verts[v];
            Vector3 vertCorr{gradientCol(base), gradientCol(base + 1), gradientCol(base + 2)};
            geom->inputVertexPositions[v] -= vertCorr;
        }

        long timeEnd = currentTimeMilliseconds();
        std::cout << "  Total time for gradient step = " << (timeEnd - timeStart) << " ms" << std::endl;
    }

    inline void printSolveInfo(size_t numNewton)
    {
        if (numNewton > 0)
        {
            std::cout << "  * With " << numNewton << " Newton constraint(s), Hs projection will require "
                      << (numNewton + 1) << " linear solves" << std::endl;
        }
        else
        {
            std::cout << "  * With no Newton constraints, Hs projection will require 1 linear solve" << std::endl;
        }
    }

    void SurfaceFlow::StepProjectedGradient()
    {
        long timeStart = currentTimeMilliseconds();
        stepCount++;
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        std::cout << "Using Hs projected gradient method..." << std::endl;
        UpdateEnergies();

        // Assemble sum of L2 differentials of all energies involved
        // (including tangent-point energy)
        Eigen::MatrixXd l2diff, gradientProj;
        l2diff.setZero(mesh->nVertices(), 3);
        gradientProj.setZero(mesh->nVertices(), 3);

        AssembleGradients(l2diff);
        double gNorm = l2diff.norm();

        std::unique_ptr<Hs::HsMetric> hs = GetHsMetric();
        hs->allowBarycenterShift = allowBarycenterShift;
        printSolveInfo(hs->newtonConstraints.size());

        Vector3 shift{0, 0, 0};
        if (allowBarycenterShift)
        {
            shift = averageOfMatrixRows(geom, mesh, l2diff);
            std::cout << "Average shift of L2 diff = " << shift << std::endl;
        }

        Hs::ProjectViaSchur<Hs::SparseInverse>(*hs, l2diff, gradientProj);

        if (allowBarycenterShift)
        {
            addShiftToMatrixRows(gradientProj, mesh->nVertices(), shift);
        }

        VertexIndices inds = mesh->getVertexIndices();

        double gProjNorm = gradientProj.norm();
        // Measure dot product of search direction with original gradient direction
        double gradDot = (l2diff.transpose() * gradientProj).trace() / (gNorm * gProjNorm);

        // Guess a step size
        // double initGuess = prevStep * 1.25;
        double initGuess = guessStepSize(gProjNorm);

        std::cout << "  * Initial step size guess = " << initGuess << std::endl;

        // Take the step using line search
        LineSearch search(mesh, geom, energies);
        double delta = search.BacktrackingLineSearch(gradientProj, initGuess, gradDot);

        if (schurConstraints.size() > 0)
        {
            Hs::ProjectSchurConstraints<Hs::SparseInverse>(*hs, 1);
        }

        if (allowBarycenterShift)
        {
            // The barycenter goes wherever it wants.
            // Assumes no pin constraints; free barycenter isn't meant to be used with pins.
        }
        else
        {
            hs->ProjectSimpleConstraints();
        }

        incrementSchurConstraints();
        geom->refreshQuantities();

        long timeEnd = currentTimeMilliseconds();
        std::cout << "  Total time for gradient step = " << (timeEnd - timeStart) << " ms" << std::endl;
    }

    void SurfaceFlow::StepProjectedGradientIterative()
    {
        long timeStart = currentTimeMilliseconds();
        stepCount++;
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        std::cout << "Using iterative Hs projected gradient method..." << std::endl;
        UpdateEnergies();

        // Assemble sum of L2 differentials of all energies involved
        // (including tangent-point energy)
        Eigen::MatrixXd l2diff, gradientProj;
        l2diff.setZero(mesh->nVertices(), 3);
        gradientProj.setZero(mesh->nVertices(), 3);
        AssembleGradients(l2diff);
        double gNorm = l2diff.norm();

        std::unique_ptr<Hs::HsMetric> hs = GetHsMetric();
        printSolveInfo(hs->newtonConstraints.size());

        Vector3 shift{0, 0, 0};
        if (allowBarycenterShift)
        {
            shift = averageOfMatrixRows(geom, mesh, l2diff);
            std::cout << "Average shift of L2 diff = " << shift << std::endl;
        }

        Hs::ProjectConstrainedHsIterativeMat(*hs, l2diff, gradientProj);

        if (allowBarycenterShift)
        {
            addShiftToMatrixRows(gradientProj, mesh->nVertices(), shift);
        }

        VertexIndices inds = mesh->getVertexIndices();

        double gProjNorm = gradientProj.norm();
        // Measure dot product of search direction with original gradient direction
        double gradDot = (l2diff.transpose() * gradientProj).trace() / (gNorm * gProjNorm);

        // Guess a step size
        // double initGuess = prevStep * 1.25;
        double initGuess = guessStepSize(gProjNorm);

        std::cout << "  * Initial step size guess = " << initGuess << std::endl;

        // Take the step using line search
        LineSearch search(mesh, geom, energies);
        double delta = search.BacktrackingLineSearch(gradientProj, initGuess, gradDot);

        // Constraint projection
        if (schurConstraints.size() > 0)
        {
            // hs->ResetSchurComplement();
            std::cout << "  Projecting Newton constraints..." << std::endl;
            Hs::ProjectSchurConstraints<Hs::IterativeInverse>(*hs, 1);
        }
        if (allowBarycenterShift)
        {
            // The barycenter goes wherever it wants.
            // Assumes no pin constraints; free barycenter isn't meant to be used with pins.
        }
        else
        {
            hs->ProjectSimpleConstraints();
        }

        incrementSchurConstraints();
        geom->refreshQuantities();

        long timeEnd = currentTimeMilliseconds();
        std::cout << "  Total time for gradient step = " << (timeEnd - timeStart) << " ms" << std::endl;
    }

    size_t SurfaceFlow::addConstraintTriplets(std::vector<Triplet> &triplets, bool includeSchur)
    {
        size_t curRow = 3 * mesh->nVertices();
        size_t nConstraintRows = 0;
        for (Constraints::SimpleProjectorConstraint *spc : simpleConstraints)
        {
            Constraints::addTripletsToSymmetric(*spc, triplets, mesh, geom, curRow);
            size_t nr = spc->nRows();
            curRow += nr;
            nConstraintRows += nr;
        }

        if (includeSchur)
        {
            for (ConstraintPack &c : schurConstraints)
            {
                Constraints::addTripletsToSymmetric(*c.constraint, triplets, mesh, geom, curRow);
                size_t nr = c.constraint->nRows();
                curRow += nr;
                nConstraintRows += nr;
            }
        }
        return nConstraintRows;
    }

    void SurfaceFlow::prefactorConstrainedLaplacian(SparseFactorization &factored, bool includeSchur)
    {
        Eigen::SparseMatrix<double> L;
        prefactorConstrainedLaplacian(L, factored, includeSchur);
    }

    void SurfaceFlow::prefactorConstrainedLaplacian(Eigen::SparseMatrix<double> &L, SparseFactorization &factored, bool includeSchur)
    {
        // Assemble triplets for the Laplacian
        std::vector<Triplet> h1Triplets, h1Triplets3x;
        H1::getTriplets(h1Triplets, mesh, geom, 1e-10);
        MatrixUtils::TripleTriplets(h1Triplets, h1Triplets3x);
        // Add constraint rows at bottom
        size_t nConstraintRows = addConstraintTriplets(h1Triplets3x, includeSchur);
        size_t dims = 3 * mesh->nVertices() + nConstraintRows;

        // Builds and factorize the Laplacian
        L.resize(dims, dims);
        L.setFromTriplets(h1Triplets3x.begin(), h1Triplets3x.end());
        factored.Compute(L);
    }

    void SurfaceFlow::StepH1ProjGrad()
    {
        long timeStart = currentTimeMilliseconds();
        stepCount++;
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        std::cout << "Using H1 projected gradient method..." << std::endl;
        UpdateEnergies();

        // Assemble sum of L2 differentials of all energies involved
        // (including tangent-point energy)
        Eigen::MatrixXd l2diff;
        l2diff.setZero(mesh->nVertices(), 3);
        AssembleGradients(l2diff);
        double gNorm = l2diff.norm();

        Vector3 shift{0, 0, 0};
        if (allowBarycenterShift)
        {
            shift = averageOfMatrixRows(geom, mesh, l2diff);
            std::cout << "Average shift of L2 diff = " << shift << std::endl;
        }

        SparseFactorization factorizedL;
        prefactorConstrainedLaplacian(factorizedL, true);
        size_t dims = factorizedL.nRows;
        std::cout << "Prefactorized" << std::endl;

        // Project the H1 gradient
        Eigen::VectorXd gradientVec;
        gradientVec.setZero(dims);
        MatrixUtils::MatrixIntoColumn(l2diff, gradientVec);

        gradientVec = factorizedL.Solve(gradientVec);
        Eigen::MatrixXd gradientProj;
        gradientProj.setZero(l2diff.rows(), l2diff.cols());
        MatrixUtils::ColumnIntoMatrix(gradientVec, gradientProj);

        if (allowBarycenterShift)
        {
            addShiftToMatrixRows(gradientProj, mesh->nVertices(), shift);
        }

        double gProjNorm = gradientProj.norm();
        // Measure dot product of search direction with original gradient direction
        double gradDot = (l2diff.transpose() * gradientProj).trace() / (gNorm * gProjNorm);
        // Guess a step size
        double initGuess = guessStepSize(gProjNorm);
        std::cout << "  * Initial step size guess = " << initGuess << std::endl;
        // Take the step using line search
        LineSearch search(mesh, geom, energies);
        double delta = search.BacktrackingLineSearch(gradientProj, initGuess, gradDot);

        // Do corrective constraint projection by reusing the H1 metric
        H1::ProjectConstraints(mesh, geom, simpleConstraints, schurConstraints, factorizedL, 1);

        incrementSchurConstraints();
        geom->refreshQuantities();

        long timeEnd = currentTimeMilliseconds();
        std::cout << "  Total time for gradient step = " << (timeEnd - timeStart) << " ms" << std::endl;
    }

    void savePositions(MeshPtr &mesh, GeomPtr &geom, Eigen::MatrixXd &positions)
    {
        positions.setZero(mesh->nVertices(), 3);

        VertexIndices inds = mesh->getVertexIndices();

        for (GCVertex v : mesh->vertices())
        {
            Vector3 pos = geom->inputVertexPositions[v];
            size_t i = inds[v];
            positions(i, 0) = pos.x;
            positions(i, 1) = pos.y;
            positions(i, 2) = pos.z;
        }
    }

    void SurfaceFlow::StepAQP(double invKappa)
    {
        long timeStart = currentTimeMilliseconds();
        if (stepCount == 0 || verticesMutated)
        {
            // Reset Nesterov memory of previous step
            savePositions(mesh, geom, prevPositions1);
            savePositions(mesh, geom, prevPositions2);
        }
        else
        {
            double theta = (1 - sqrt(invKappa)) / (1 + sqrt(invKappa));
            // 1. Nesterov step
            Eigen::MatrixXd y_n = (1 + theta) * prevPositions1 - theta * prevPositions2;
            for (GCVertex v : mesh->vertices())
            {
                Vector3 pos = MatrixUtils::GetRowAsVector3(y_n, v.getIndex());
                geom->inputVertexPositions[v] = pos;
            }
            geom->refreshQuantities();
        }
        stepCount++;

        // 2. H1 gradient step following the Nesterov step
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        std::cout << "Using AQP..." << std::endl;
        UpdateEnergies();

        // Assemble sum of L2 differentials of all energies involved
        // (including tangent-point energy)
        Eigen::MatrixXd l2diff;
        l2diff.setZero(mesh->nVertices(), 3);
        AssembleGradients(l2diff);
        double gNorm = l2diff.norm();

        Vector3 shift{0, 0, 0};
        if (allowBarycenterShift)
        {
            shift = averageOfMatrixRows(geom, mesh, l2diff);
            std::cout << "Average shift of L2 diff = " << shift << std::endl;
        }

        SparseFactorization factorizedL;
        // Only use "simple" positional constraints (Nesterov would break hard constraints anyway)
        prefactorConstrainedLaplacian(factorizedL, false);
        size_t dims = factorizedL.nRows;

        // Project the H1 gradient
        Eigen::VectorXd gradientVec;
        gradientVec.setZero(dims);
        MatrixUtils::MatrixIntoColumn(l2diff, gradientVec);

        gradientVec = factorizedL.Solve(gradientVec);
        Eigen::MatrixXd gradientProj;
        gradientProj.setZero(l2diff.rows(), l2diff.cols());
        MatrixUtils::ColumnIntoMatrix(gradientVec, gradientProj);

        if (allowBarycenterShift)
        {
            addShiftToMatrixRows(gradientProj, mesh->nVertices(), shift);
        }

        double gProjNorm = gradientProj.norm();
        // Measure dot product of search direction with original gradient direction
        double gradDot = (l2diff.transpose() * gradientProj).trace() / (gNorm * gProjNorm);
        // Guess a step size
        double initGuess = guessStepSize(gProjNorm);
        std::cout << "  * Initial step size guess = " << initGuess << std::endl;
        // Take the step using line search
        LineSearch search(mesh, geom, energies);
        double delta = search.BacktrackingLineSearch(gradientProj, initGuess, gradDot);

        // Make sure pins don't drift
        for (Constraints::SimpleProjectorConstraint *spc : simpleConstraints)
        {
            spc->ProjectConstraint(mesh, geom);
        }
        // Save previous positions
        prevPositions2 = prevPositions1;
        savePositions(mesh, geom, prevPositions1);
        geom->refreshQuantities();

        long timeEnd = currentTimeMilliseconds();
        std::cout << "  Total time for gradient step = " << (timeEnd - timeStart) << " ms" << std::endl;
    }

    void SurfaceFlow::StepH1LBFGS()
    {
        long timeStart = currentTimeMilliseconds();

        stepCount++;
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        std::cout << "Using H1 L-BFGS..." << std::endl;

        if (!lbfgs)
        {
            lbfgs = new H1_LBFGS(20, simpleConstraints);
        }

        if (verticesMutated)
        {
            std::cout << "  * Vertices were mutated; resetting memory" << std::endl;
            lbfgs->ResetMemory();
        }

        UpdateEnergies();
        // Assemble sum of L2 differentials of all energies involved
        // (including tangent-point energy)
        Eigen::MatrixXd l2diff, projected, positions;
        l2diff.setZero(mesh->nVertices(), 3);
        positions.setZero(mesh->nVertices(), 3);
        projected.setZero(mesh->nVertices(), 3);
        AssembleGradients(l2diff);
        double gNorm = l2diff.norm();

        Eigen::VectorXd l2diffvec(3 * mesh->nVertices());
        MatrixUtils::MatrixIntoColumn(l2diff, l2diffvec);

        savePositions(mesh, geom, positions);
        Eigen::VectorXd posvec(3 * mesh->nVertices());
        MatrixUtils::MatrixIntoColumn(positions, posvec);

        // Do the L-BFGS update with current position and gradient
        lbfgs->SetUpInnerProduct(mesh, geom);
        lbfgs->UpdateDirection(posvec, l2diffvec);
        double gProjNorm = lbfgs->direction().norm();

        MatrixUtils::ColumnIntoMatrix(lbfgs->direction(), projected);
        double gradDot = (l2diffvec.dot(lbfgs->direction())) / (gNorm * gProjNorm);
        std::cout << "  * Dot product = " << gradDot << std::endl;

        LineSearch search(mesh, geom, energies);
        // Take the step using line search
        double initGuess = guessStepSize(gProjNorm);
        double delta = search.BacktrackingLineSearch(projected, initGuess, fmax(0, gradDot));

        if (gradDot < 0)
        {
            std::cout << "  * Negative dot product; resetting memory" << std::endl;
            lbfgs->ResetMemory();
        }

        // Make sure pins don't drift
        for (Constraints::SimpleProjectorConstraint *spc : simpleConstraints)
        {
            spc->ProjectConstraint(mesh, geom);
        }
        geom->refreshQuantities();

        long timeEnd = currentTimeMilliseconds();
        std::cout << "  Total time for gradient step = " << (timeEnd - timeStart) << " ms" << std::endl;
    }

    void SurfaceFlow::StepBQN()
    {
        long timeStart = currentTimeMilliseconds();

        stepCount++;
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        std::cout << "Using blended L-BFGS (BQN)..." << std::endl;

        if (!lbfgs)
        {
            // B = (total area) ^ (2(d-1) / d) for d = 3
            // (equation 13 of BCQN / Zhu et al.)
            double b = pow(totalArea(geom, mesh), 4.0 / 3.0);
            lbfgs = new BQN_LBFGS(20, simpleConstraints, b);
        }

        if (verticesMutated)
        {
            std::cout << "  * Vertices were mutated; resetting memory" << std::endl;
            lbfgs->ResetMemory();
        }

        UpdateEnergies();
        // Assemble sum of L2 differentials of all energies involved
        // (including tangent-point energy)
        Eigen::MatrixXd l2diff, projected, positions;
        l2diff.setZero(mesh->nVertices(), 3);
        positions.setZero(mesh->nVertices(), 3);
        projected.setZero(mesh->nVertices(), 3);
        AssembleGradients(l2diff);
        double gNorm = l2diff.norm();

        Eigen::VectorXd l2diffvec(3 * mesh->nVertices());
        MatrixUtils::MatrixIntoColumn(l2diff, l2diffvec);

        savePositions(mesh, geom, positions);
        Eigen::VectorXd posvec(3 * mesh->nVertices());
        MatrixUtils::MatrixIntoColumn(positions, posvec);

        // Do the L-BFGS update with current position and gradient
        lbfgs->SetUpInnerProduct(mesh, geom);
        lbfgs->UpdateDirection(posvec, l2diffvec);
        double gProjNorm = lbfgs->direction().norm();

        MatrixUtils::ColumnIntoMatrix(lbfgs->direction(), projected);
        double gradDot = (l2diffvec.dot(lbfgs->direction())) / (gNorm * gProjNorm);
        std::cout << "  * Dot product = " << gradDot << std::endl;

        LineSearch search(mesh, geom, energies);
        // Take the step using line search
        double initGuess = guessStepSize(gProjNorm);
        double delta = search.BacktrackingLineSearch(projected, initGuess, fmax(0, gradDot));

        if (gradDot < 0)
        {
            std::cout << "  * Negative dot product; resetting memory" << std::endl;
            lbfgs->ResetMemory();
        }

        // Make sure pins don't drift
        for (Constraints::SimpleProjectorConstraint *spc : simpleConstraints)
        {
            spc->ProjectConstraint(mesh, geom);
        }
        geom->refreshQuantities();

        long timeEnd = currentTimeMilliseconds();
        std::cout << "  Total time for gradient step = " << (timeEnd - timeStart) << " ms" << std::endl;
    }

    void SurfaceFlow::RecenterMesh()
    {
        Vector3 center = meshBarycenter(geom, mesh);
        translateMesh(geom, mesh, origBarycenter - center);
    }

    void SurfaceFlow::ResetAllConstraints()
    {
        for (ConstraintPack &p : schurConstraints)
        {
            p.constraint->ResetFunction(mesh, geom);
        }
        for (Constraints::SimpleProjectorConstraint *c : simpleConstraints)
        {
            c->ResetFunction(mesh, geom);
        }
    }

    void SurfaceFlow::ResetAllPotentials()
    {
        for (SurfaceEnergy *energy : energies)
        {
            energy->ResetTargets();
        }
    }

    SurfaceEnergy *SurfaceFlow::BaseEnergy()
    {
        return energies[0];
    }

} // namespace rsurfaces