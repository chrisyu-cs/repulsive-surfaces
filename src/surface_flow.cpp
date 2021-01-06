#include "surface_flow.h"
#include "fractional_laplacian.h"
#include "helpers.h"

#include "sobolev/h1.h"
#include "sobolev/hs.h"
#include "sobolev/hs_schur.h"
#include "sobolev/hs_iterative.h"
#include "sobolev/constraints.h"
#include "spatial/convolution.h"
#include "energy/barnes_hut_tpe_6d.h"

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

        ncg = 0;
    }

    void SurfaceFlow::AddAdditionalEnergy(SurfaceEnergy *extraEnergy)
    {
        energies.push_back(extraEnergy);
    }

    void SurfaceFlow::StepNaive(double t)
    {
        stepCount++;
        double energyBefore = GetEnergyValue(energies);

        Eigen::MatrixXd gradient;
        gradient.setZero(mesh->nVertices(), 3);
        AddGradientsToMatrix(energies, gradient);
        surface::VertexData<size_t> indices = mesh->getVertexIndices();

        for (GCVertex v : mesh->vertices())
        {
            Vector3 grad_v = GetRow(gradient, indices[v]);
            geom->inputVertexPositions[v] -= grad_v * t;
        }

        double energyAfter = GetEnergyValue(energies);

        std::cout << "Energy: " << energyBefore << " -> " << energyAfter << std::endl;
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

    double SurfaceFlow::evaluateEnergy()
    {
        return GetEnergyValue(energies);
    }

    void SurfaceFlow::StepNCG()
    {
        long timeStart = currentTimeMilliseconds();

        stepCount++;
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        std::cout << "Using Hs NCG method..." << std::endl;
        UpdateEnergies();

        // Grab the tangent-point energy specifically
        BarnesHutTPEnergy6D *bhEnergy = dynamic_cast<BarnesHutTPEnergy6D *>(energies[0]);

        // Get the current L2 differential of energies
        Eigen::MatrixXd l2diff, hsGrad;
        l2diff.setZero(mesh->nVertices(), 3);
        hsGrad = l2diff;
        AssembleGradients(l2diff);
        double l2DiffNorm = l2diff.norm();

        std::cout << "l2 diff norm (outside) = " << l2DiffNorm << std::endl;

        if (!ncg)
        {
            ncg = new Hs::HsNCG(bhEnergy, simpleConstraints);
        }

        std::unique_ptr<Hs::HsMetric> hs = GetHsMetric();

        double hsGradNormBefore = ncg->UpdateConjugateDir(l2diff, *hs);
        double gProjNorm = ncg->direction().norm();
        double initGuess = guessStepSize(gProjNorm);

        double gradDot = fabs((l2diff.transpose() * ncg->direction()).trace()) / (l2DiffNorm * gProjNorm);

        // Take a line search step
        LineSearch search(mesh, geom, energies);
        search.BacktrackingLineSearch(ncg->direction(), initGuess, gradDot, false);

        /*
        // Check Wolfe condition
        l2diff.setZero();
        AddGradientsToMatrix(energies, l2diff);
        // double l2DiffNormAfter = l2diff.norm();
        Hs::HsMetric hs(bhEnergy, simpleConstraints);
        hs.InvertMetricMat(l2diff, hsGrad);
        double hsGradNormAfter = (l2diff.transpose() * hsGrad).trace();

        if (hsGradNormAfter >= hsGradNormBefore)
        {
            std::cout << "Wolfe condition failed; resetting NCG memory" << std::endl;
            ncg->ResetMemory();
        }
        */

        std::cout << "  Mesh total volume = " << totalVolume(geom, mesh) << std::endl;
        std::cout << "  Mesh total area = " << totalArea(geom, mesh) << std::endl;
        geom->refreshQuantities();

        long timeEnd = currentTimeMilliseconds();
        std::cout << "  Total time for gradient step = " << (timeEnd - timeStart) << " ms" << std::endl;
    }

    void SurfaceFlow::AssembleGradients(Eigen::MatrixXd &dest)
    {
        AddGradientsToMatrix(energies, dest);
    }

    std::unique_ptr<Hs::HsMetric> SurfaceFlow::GetHsMetric()
    {
        BarnesHutTPEnergy6D *bhEnergy = dynamic_cast<BarnesHutTPEnergy6D *>(energies[0]);
        std::unique_ptr<Hs::HsMetric> hs(new Hs::HsMetric(bhEnergy, simpleConstraints, schurConstraints));
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
        size_t curRow = 0;

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
            geom->inputVertexPositions[v] += vertCorr;
        }

        long timeEnd = currentTimeMilliseconds();
        std::cout << "  Total time for gradient step = " << (timeEnd - timeStart) << " ms" << std::endl;
    }

    inline void printSolveInfo(size_t numNewton)
    {
        if (numNewton > 0)
        {
            std::cout << "  * With " << numNewton << " Newton constraint(s), Hs projection will require "
                      << (numNewton + 2) << " linear solves" << std::endl;
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
            std::cout << "Shifting barycenter by " << shift * delta << std::endl;
            hs->shiftBarycenterConstraint(-delta * shift);
        }
        hs->ProjectSimpleConstraints();

        incrementSchurConstraints();

        std::cout << "  Mesh total volume = " << totalVolume(geom, mesh) << std::endl;
        std::cout << "  Mesh total area = " << totalArea(geom, mesh) << std::endl;
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
            std::cout << "Shifting barycenter by " << shift * delta << std::endl;
            hs->shiftBarycenterConstraint(-delta * shift);
        }
        hs->ProjectSimpleConstraints();

        incrementSchurConstraints();

        std::cout << "  Mesh total volume = " << totalVolume(geom, mesh) << std::endl;
        std::cout << "  Mesh total area = " << totalArea(geom, mesh) << std::endl;
        geom->refreshQuantities();

        long timeEnd = currentTimeMilliseconds();
        std::cout << "  Total time for gradient step = " << (timeEnd - timeStart) << " ms" << std::endl;
    }

    size_t SurfaceFlow::addConstraintTriplets(std::vector<Triplet> &triplets)
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

        for (ConstraintPack &c : schurConstraints)
        {
            Constraints::addTripletsToSymmetric(*c.constraint, triplets, mesh, geom, curRow);
            size_t nr = c.constraint->nRows();
            curRow += nr;
            nConstraintRows += nr;
        }
        return nConstraintRows;
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

        // Assemble triplets for the Laplacian
        std::vector<Triplet> h1Triplets, h1Triplets3x;
        H1::getTriplets(h1Triplets, mesh, geom, 1e-10);
        MatrixUtils::TripleTriplets(h1Triplets, h1Triplets3x);
        // Add constraint rows at bottom
        size_t nConstraintRows = addConstraintTriplets(h1Triplets3x);
        size_t dims = 3 * mesh->nVertices() + nConstraintRows;

        // Builds and factorize the Laplacian
        Eigen::SparseMatrix<double> L(dims, dims);
        L.setFromTriplets(h1Triplets3x.begin(), h1Triplets3x.end());
        SparseFactorization factorizedL;
        factorizedL.Compute(L);

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

        // Do corrective constraint projection by reusing the H1 metric
        H1::ProjectConstraints(mesh, geom, simpleConstraints, schurConstraints, factorizedL, 1);

        incrementSchurConstraints();

        std::cout << "  Mesh total volume = " << totalVolume(geom, mesh) << std::endl;
        std::cout << "  Mesh total area = " << totalArea(geom, mesh) << std::endl;
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