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
        RecenterMesh();
    }

    void SurfaceFlow::AddAdditionalEnergy(SurfaceEnergy *extraEnergy)
    {
        energies.push_back(extraEnergy);
    }

    void SurfaceFlow::StepNaive(double t)
    {
        stepCount++;
        double energyBefore = GetEnergyValue();

        Eigen::MatrixXd gradient;
        gradient.setZero(mesh->nVertices(), 3);
        AddGradientToMatrix(gradient);
        surface::VertexData<size_t> indices = mesh->getVertexIndices();

        for (GCVertex v : mesh->vertices())
        {
            Vector3 grad_v = GetRow(gradient, indices[v]);
            geom->inputVertexPositions[v] -= grad_v * t;
        }

        double energyAfter = GetEnergyValue();

        std::cout << "Energy: " << energyBefore << " -> " << energyAfter << std::endl;
    }

    void SurfaceFlow::UpdateEnergies()
    {
        for (SurfaceEnergy *energy : energies)
        {
            energy->Update();
        }
    }

    double SurfaceFlow::GetEnergyValue()
    {
        double sum = 0;
        for (SurfaceEnergy *energy : energies)
        {
            sum += energy->Value();
        }
        return sum;
    }

    void SurfaceFlow::AddGradientToMatrix(Eigen::MatrixXd &gradient)
    {
        int i = 0;
        for (SurfaceEnergy *energy : energies)
        {
            i++;
            long before = currentTimeMilliseconds();
            energy->Differential(gradient);
            long after = currentTimeMilliseconds();
            std::cout << "  * Energy " << i << " gradient = " << (after - before) << " ms" << std::endl;
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
        double energyBefore = GetEnergyValue();
        std::cout << "Initial energy = " << energyBefore << std::endl;
        long timeStart = currentTimeMilliseconds();

        std::cout << "  * Energy evaluation: " << (timeStart - timeEnergy) << " ms" << std::endl;

        // Assemble sum of gradients of all energies involved
        // (including tangent-point energy)
        Eigen::MatrixXd gradient, gradientProj;
        gradient.setZero(mesh->nVertices(), 3);
        gradientProj.setZero(mesh->nVertices(), 3);
        AddGradientToMatrix(gradient);
        double gNorm = gradient.norm();

        long timeDiff = currentTimeMilliseconds();
        std::cout << "  * Gradient assembly: " << (timeDiff - timeStart) << " ms (norm = " << gNorm << ")" << std::endl;

        Hs::HsMetric hs(bhEnergy, simpleConstraints);

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
        double initGuess = prevStep * 1.25;
        if (prevStep <= LS_STEP_THRESHOLD)
        {
            initGuess = (gProjNorm < 1) ? 1.0 / sqrt(gProjNorm) : 1.0 / gProjNorm;
        }
        std::cout << "  * Initial step size guess = " << initGuess << std::endl;

        // Take the step using line search
        LineSearchStep(gradientProj, initGuess, gradDot);
        long timeLS = currentTimeMilliseconds();
        std::cout << "  * Line search: " << (timeLS - timeProject) << " ms" << std::endl;

        double energyBeforeBackproj = GetEnergyValue();

        long timeBackproj = currentTimeMilliseconds();
        
        // H1::ProjectConstraints(mesh, geom, schurConstraints, simpleConstraints, 2);

        if (schurConstraints.size() > 0)
        {
            hs.ProjectSchurConstraints(schurConstraints, comp, 1);
        }
        hs.ProjectSimpleConstraints();

        std::cout << "  Barycenter = " << meshBarycenter(geom, mesh) << std::endl;
        

        long timeEnd = currentTimeMilliseconds();

        std::cout << "  * Post-processing: " << (timeEnd - timeBackproj) << " ms" << std::endl;
        std::cout << "  Total time: " << (timeEnd - timeStart) << " ms" << std::endl;
        std::cout << "  Energy: " << energyBefore << " -> " << energyBeforeBackproj << std::endl;

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

    void SurfaceFlow::SaveCurrentPositions()
    {
        surface::VertexData<size_t> indices = mesh->getVertexIndices();
        origPositions.setZero(mesh->nVertices(), 3);
        for (GCVertex v : mesh->vertices())
        {
            // Copy the vertex positions of each vertex into
            // the corresponding row
            Vector3 pos_v = geom->inputVertexPositions[v];
            size_t ind_v = indices[v];
            origPositions(ind_v, 0) = pos_v.x;
            origPositions(ind_v, 1) = pos_v.y;
            origPositions(ind_v, 2) = pos_v.z;
        }
    }

    void SurfaceFlow::RestorePositions()
    {
        surface::VertexData<size_t> indices = mesh->getVertexIndices();
        for (GCVertex v : mesh->vertices())
        {
            // Copy the positions in each row of origPositions
            // back to each vertex
            size_t ind_v = indices[v];
            Vector3 pos_v = GetRow(origPositions, ind_v);
            geom->inputVertexPositions[v] = pos_v;
        }

        if (energies[0]->GetBVH())
        {
            geom->refreshQuantities();
            energies[0]->GetBVH()->recomputeCentersOfMass(mesh, geom);
        }
    }

    void SurfaceFlow::SetGradientStep(Eigen::MatrixXd &gradient, double delta)
    {
        surface::VertexData<size_t> indices = mesh->getVertexIndices();
        for (GCVertex v : mesh->vertices())
        {
            // Set the position of each vertex to be the sum of
            // origPositions + delta * gradient
            size_t ind_v = indices[v];
            Vector3 pos_v = GetRow(origPositions, ind_v);
            Vector3 grad_v = GetRow(gradient, ind_v);
            geom->inputVertexPositions[v] = pos_v - delta * grad_v;
        }

        if (energies[0]->GetBVH())
        {
            geom->refreshQuantities();
            energies[0]->GetBVH()->recomputeCentersOfMass(mesh, geom);
        }
    }

    double SurfaceFlow::LineSearchStep(Eigen::MatrixXd &gradient, double initGuess, double gradDot)
    {
        double delta = initGuess;
        SaveCurrentPositions();

        // Gather some initial data
        double initialEnergy = GetEnergyValue();
        double gradNorm = gradient.norm();
        int numBacktracks = 0;
        double sigma = 0.01;
        double nextEnergy = initialEnergy;

        if (gradNorm < 1e-10)
        {
            std::cout << "* Gradient is very close to zero" << std::endl;
            return 0;
        }

        while (delta > LS_STEP_THRESHOLD)
        {
            // Take the gradient step
            SetGradientStep(gradient, delta);
            nextEnergy = GetEnergyValue();
            double decrease = initialEnergy - nextEnergy;
            double targetDecrease = sigma * delta * gradNorm * gradDot;

            if (decrease < targetDecrease)
            {
                delta /= 2;
                numBacktracks++;
            }
            // Otherwise, accept the current step.
            else
            {
                prevStep = delta;
                break;
            }
        }

        if (delta <= LS_STEP_THRESHOLD)
        {
            std::cout << "* Failed to find a non-trivial step after " << numBacktracks << " backtracks" << std::endl;
            prevStep = 0;
            // Restore initial positions if step size goes to 0
            RestorePositions();
            return 0;
        }
        else
        {
            std::cout << "  * Took step of size " << delta << " after " << numBacktracks << " backtracks" << std::endl;
            return delta;
        }
    }

} // namespace rsurfaces