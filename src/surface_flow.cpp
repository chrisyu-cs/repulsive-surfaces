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

    void SurfaceFlow::StepFractionalSobolev(Preserve p_what)
    {

        stepCount++;
        std::cout << "=== Iteration " << stepCount << " ===" << std::endl;
        energy->Update();
        double initArea = 0;

        if (p_what == Preserve::Area)
        {
            initArea = totalArea(geom, mesh);
            std::cout << "  * Total surface area = " << initArea << std::endl;
        }
        else if (p_what == Preserve::Volume)
        {
            initArea = totalVolume(geom, mesh);
            std::cout << "  * Total volume = " << initArea << std::endl;
        }

        double energyBefore = energy->Value();
        long timeStart = currentTimeMilliseconds();
        Eigen::MatrixXd gradient, gradientProj;
        gradient.setZero(mesh->nVertices(), 3);
        gradientProj.setZero(mesh->nVertices(), 3);

        energy->Differential(gradient);
        double gNorm = gradient.norm();

        long timeDiff = currentTimeMilliseconds();
        std::cout << "  * Gradient assembly: " << (timeDiff - timeStart) << " ms (norm = " << gNorm << ")" << std::endl;

        Vector2 alpha_beta = energy->GetExponents();

        // H1::ProjectGradient(gradient, gradientProj, mesh, geom, false);
        Hs::ProjectGradient(gradient, gradientProj, alpha_beta.x, alpha_beta.y, mesh, geom);
        // Hs::ProjectViaSparse(gradient, gradientProj, alpha_beta.x, alpha_beta.y, mesh, geom, energy->GetBVH());
        // gradientProj = gradient;
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

        RecenterMesh();
        if (p_what == Preserve::Area)
        {
            RescaleToPreserveArea(initArea);
        }
        if (p_what == Preserve::Volume)
        {
            RescaleToPreserveVolume(initArea);
        }

        long timeEnd = currentTimeMilliseconds();
        double energyAfter = energy->Value();

        std::cout << "  * Post-processing: " << (timeEnd - timeLS) << " ms" << std::endl;

        std::cout << "  Total time: " << (timeEnd - timeStart) << " ms" << std::endl;
        std::cout << "  Energy: " << energyBefore << " -> " << energyAfter << std::endl;
    }

    void SurfaceFlow::RecenterMesh()
    {
        Vector3 center = meshBarycenter(geom, mesh);
        translateMesh(geom, mesh, -center);
    }

    void SurfaceFlow::RescaleToPreserveArea(double area)
    {
        double currentArea = totalArea(geom, mesh);
        std::cout << "  * New area = " << currentArea << std::endl;
        double reqScale = sqrt(area / currentArea);
        scaleMesh(geom, mesh, reqScale, Vector3{0, 0, 0});
    }

    void SurfaceFlow::RescaleToPreserveVolume(double volume)
    {
        double currentVolume = totalVolume(geom, mesh);
        std::cout << "  * New volume = " << currentVolume << std::endl;
        double reqScale = pow(volume / currentVolume, 1. / 3.);
        scaleMesh(geom, mesh, reqScale, Vector3{0, 0, 0});
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