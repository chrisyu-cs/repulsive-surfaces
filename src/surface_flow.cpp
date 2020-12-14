#include "surface_flow.h"
#include "fractional_laplacian.h"
#include "helpers.h"

#include "sobolev/h1.h"
#include "sobolev/hs.h"
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

    void SurfaceFlow::StepFractionalSobolev()
    {
        stepCount++;
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        UpdateEnergies();

        // Grab the tangent-point energy specifically
        BarnesHutTPEnergy6D *bhEnergy = dynamic_cast<BarnesHutTPEnergy6D *>(energies[0]);

        // Measure the energy at the start of the timestep -- just for
        // diagnostic purposes
        long timeEnergy = currentTimeMilliseconds();
        double energyBefore = GetEnergyValue(energies);
        std::cout << "Initial energy = " << energyBefore << std::endl;
        long timeStart = currentTimeMilliseconds();

        std::cout << "  * Energy evaluation: " << (timeStart - timeEnergy) << " ms" << std::endl;

        // Assemble sum of gradients of all energies involved
        // (including tangent-point energy)
        Eigen::MatrixXd gradient, gradientProj;
        gradient.setZero(mesh->nVertices(), 3);
        gradientProj.setZero(mesh->nVertices(), 3);

        AddGradientsToMatrix(energies, gradient);
        double gNorm = gradient.norm();

        long timeDiff = currentTimeMilliseconds();
        std::cout << "  * Gradient assembly: " << (timeDiff - timeStart) << " ms (norm = " << gNorm << ")" << std::endl;

        Hs::HsMetric hs(bhEnergy, simpleConstraints);
        LineSearch search(mesh, geom, energies);

        // Schur complement will be reused in multiple steps
        Hs::SchurComplement comp;
        if (schurConstraints.size() > 0)
        {
            hs.GetSchurComplement(schurConstraints, comp);
            hs.ProjectViaSchur(gradient, gradientProj, comp);
        }
        else
        {
            hs.ProjectViaSparseMat(gradient, gradientProj);
        }

        VertexIndices inds = mesh->getVertexIndices();

        long timeProject = currentTimeMilliseconds();
        double gProjNorm = gradientProj.norm();
        std::cout << "  * Gradient projection: " << (timeProject - timeDiff) << " ms (norm = " << gProjNorm << ")" << std::endl;
        // Measure dot product of search direction with original gradient direction
        double gradDot = (gradient.transpose() * gradientProj).trace() / (gNorm * gProjNorm);

        // Guess a step size
        // double initGuess = prevStep * 1.25;
        double initGuess = (gProjNorm < 1) ? 1.0 / sqrt(gProjNorm) : 1.0 / gProjNorm;
        initGuess *= 2;

        std::cout << "  * Initial step size guess = " << initGuess << std::endl;

        // Take the step using line search
        search.BacktrackingLineSearch(gradientProj, initGuess, gradDot);
        long timeLS = currentTimeMilliseconds();
        std::cout << "  * Line search: " << (timeLS - timeProject) << " ms" << std::endl;

        long timeBackproj = currentTimeMilliseconds();
        
        if (schurConstraints.size() > 0)
        {
            hs.ProjectSchurConstraints(schurConstraints, comp, 1);
        }
        hs.ProjectSimpleConstraints();

        std::cout << "  Barycenter = " << meshBarycenter(geom, mesh) << std::endl;
        

        long timeEnd = currentTimeMilliseconds();

        std::cout << "  * Post-processing: " << (timeEnd - timeBackproj) << " ms" << std::endl;
        std::cout << "  Total time: " << (timeEnd - timeStart) << " ms" << std::endl;

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