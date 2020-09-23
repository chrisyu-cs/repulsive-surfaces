#include "surface_flow.h"
#include "fractional_laplacian.h"
#include "helpers.h"

#include "sobolev/h1.h"
#include "sobolev/hs.h"
#include "sobolev/constraints.h"
#include "spatial/convolution.h"

#include <Eigen/SparseCholesky>

namespace rsurfaces
{

    SurfaceFlow::SurfaceFlow(SurfaceEnergy *energy_)
    {
        energy = energy_;
        mesh = energy->GetMesh();
        geom = energy->GetGeom();
        stepCount = 0;

        RecenterMesh();
    }

    void SurfaceFlow::StepNaive(double t)
    {
        stepCount++;
        double energyBefore = energy->Value();
        Eigen::MatrixXd gradient;
        gradient.setZero(mesh->nVertices(), 3);
        energy->Differential(gradient);
        surface::VertexData<size_t> indices = mesh->getVertexIndices();

        for (GCVertex v : mesh->vertices())
        {
            Vector3 grad_v = GetRow(gradient, indices[v]);
            geom->inputVertexPositions[v] -= grad_v * t;
        }

        double energyAfter = energy->Value();

        std::cout << "Energy: " << energyBefore << " -> " << energyAfter << std::endl;
    }

    void SurfaceFlow::StepFractionalSobolev()
    {
        stepCount++;
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        energy->Update();
        double initArea = totalArea(geom, mesh);
        double initVolume = totalVolume(geom, mesh);

        long timeEnergy = currentTimeMilliseconds();
        double energyBefore = energy->Value();
        long timeStart = currentTimeMilliseconds();

        std::cout << "  * Energy evaluation: " << (timeStart - timeEnergy) << " ms" << std::endl;

        Eigen::MatrixXd gradient, gradientProj;
        gradient.setZero(mesh->nVertices(), 3);
        gradientProj.setZero(mesh->nVertices(), 3);

        energy->Differential(gradient);
        double gNorm = gradient.norm();

        long timeDiff = currentTimeMilliseconds();
        std::cout << "  * Gradient assembly: " << (timeDiff - timeStart) << " ms (norm = " << gNorm << ")" << std::endl;

        // Set up some data that will be reused in multiple steps:
        // Schur complement, and sparse factorization (of Laplacian)
        Hs::SchurComplement comp;
        Hs::SparseFactorization factor;
        // Create an empty BCT pointer; this will be initialized in the first
        // Schur complement function, and reused for the rest of this timestep
        BlockClusterTree *bct = 0;
        Hs::GetSchurComplement(constraints, energy, comp, bct, factor);
        Hs::ProjectViaSchur(comp, gradient, gradientProj, energy, bct, factor);

        long timeProject = currentTimeMilliseconds();
        double gProjNorm = gradientProj.norm();
        std::cout << "  * Gradient projection: " << (timeProject - timeDiff) << " ms (norm = " << gProjNorm << ")" << std::endl;
        double gradDot = (gradient.transpose() * gradientProj).trace() / (gNorm * gProjNorm);
        std::cout << "  * Dot product of search directions = " << gradDot << std::endl;

        if (gradDot < 0)
        {
            gradientProj = -gradientProj;
            gradDot = -gradDot;
            std::cout << "  * Dot product negative; negating search direction" << std::endl;
        }

        // Guess a step size
        double initGuess = prevStep * 1.25;
        if (prevStep <= LS_STEP_THRESHOLD)
        {
            initGuess = (gProjNorm < 1) ? 1.0 / sqrt(gProjNorm) : 1.0 / gProjNorm;
        }
        std::cout << "  * Initial step size guess = " << initGuess << std::endl;

        // Take the step
        LineSearchStep(gradientProj, initGuess, gradDot);
        long timeLS = currentTimeMilliseconds();
        std::cout << "  * Line search: " << (timeLS - timeProject) << " ms" << std::endl;

        double energyBeforeBackproj = energy->Value();

        // Project onto constraint manifold using Schur complement
        long timeBackproj = currentTimeMilliseconds();
        Hs::BackprojectViaSchur(constraints, comp, energy, bct, factor);
        // Fix barycenter drift
        RecenterMesh();
        long timeEnd = currentTimeMilliseconds();

        double energyAfter = energy->Value();
        // Done with BCT, so clean it up
        delete bct;

        std::cout << "  * Post-processing: " << (timeEnd - timeBackproj) << " ms" << std::endl;

        std::cout << "  Total time: " << (timeEnd - timeStart) << " ms" << std::endl;
        std::cout << "  Energy: " << energyBefore << " -> " << energyBeforeBackproj << " -> " << energyAfter << std::endl;

        double finalArea = totalArea(geom, mesh);
        double finalVolume = totalVolume(geom, mesh);

        std::cout << "Area:   " << initArea << " -> " << finalArea << std::endl;
        std::cout << "Volume: " << initVolume << " -> " << finalVolume << std::endl;
        geom->refreshQuantities();
    }

    void SurfaceFlow::RecenterMesh()
    {
        Vector3 center = meshBarycenter(geom, mesh);
        translateMesh(geom, mesh, -center);
    }

    SurfaceEnergy *SurfaceFlow::BaseEnergy()
    {
        return energy;
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

        if (energy->GetBVH())
        {
            geom->refreshQuantities();
            energy->GetBVH()->recomputeCentersOfMass(mesh, geom);
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

        if (energy->GetBVH())
        {
            geom->refreshQuantities();
            energy->GetBVH()->recomputeCentersOfMass(mesh, geom);
        }
    }

    double SurfaceFlow::LineSearchStep(Eigen::MatrixXd &gradient, double initGuess, double gradDot)
    {
        double delta = initGuess;
        SaveCurrentPositions();

        // Gather some initial data
        double initialEnergy = energy->Value();
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
            nextEnergy = energy->Value();
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