#include "surface_flow.h"
#include "fractional_laplacian.h"
#include "helpers.h"

#include "sobolev/h1.h"
#include "sobolev/hs.h"
#include "sobolev/hs_schur.h"
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

        double initGuess = (gProjNorm < 1) ? 1.0 / sqrt(gProjNorm) : 1.0 / gProjNorm;
        initGuess *= 2;
        return initGuess;
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

        double hsGradNormBefore = ncg->UpdateConjugateDir(l2diff);
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

    std::unique_ptr<Hs::HsMetric> SurfaceFlow::GetMetric()
    {
        BarnesHutTPEnergy6D *bhEnergy = dynamic_cast<BarnesHutTPEnergy6D *>(energies[0]);
        return std::unique_ptr<Hs::HsMetric>(new Hs::HsMetric(bhEnergy, simpleConstraints));
    }


    void SurfaceFlow::StepProjectedGradient()
    {
        long timeStart = currentTimeMilliseconds();
        stepCount++;
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        std::cout << "Using Hs projected gradient method..." << std::endl;
        UpdateEnergies();

        // Grab the tangent-point energy specifically

        // Assemble sum of L2 differentials of all energies involved
        // (including tangent-point energy)
        Eigen::MatrixXd l2diff, gradientProj;
        l2diff.setZero(mesh->nVertices(), 3);
        gradientProj.setZero(mesh->nVertices(), 3);

        AssembleGradients(l2diff);
        double gNorm = l2diff.norm();

        std::unique_ptr<Hs::HsMetric> hs = GetMetric();

        // Schur complement will be reused in multiple steps
        Hs::SchurComplement comp;
        if (schurConstraints.size() > 0)
        {
            GetSchurComplement(*hs, schurConstraints, comp);
            ProjectViaSchur(*hs, l2diff, gradientProj, comp);
        }
        else
        {
            hs->InvertMetricMat(l2diff, gradientProj);
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
        search.BacktrackingLineSearch(gradientProj, initGuess, gradDot);

        if (schurConstraints.size() > 0)
        {
            ProjectSchurConstraints(*hs, schurConstraints, comp, 1);
        }
        hs->ProjectSimpleConstraints();

        for (ConstraintPack &c : schurConstraints)
        {
            if (c.iterationsLeft > 0)
            {
                c.iterationsLeft--;
                c.constraint->incrementTargetValue(c.stepSize);
            }
        }

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