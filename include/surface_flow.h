#pragma once

#include "rsurface_types.h"
#include "sobolev/constraints.h"

namespace rsurfaces
{

    class SurfaceFlow
    {
    public:
        SurfaceFlow(SurfaceEnergy *energy_);
        void StepNaive(double t);
        void StepFractionalSobolev();
        SurfaceEnergy *BaseEnergy();
        double LineSearchStep(Eigen::MatrixXd &gradient, double initGuess, double gradDot);

        void RescaleToPreserveArea(double area);
        void RescaleToPreserveVolume(double area);
        void RecenterMesh();
        const double LS_STEP_THRESHOLD = 1e-20;

        template <typename Constraint>
        void addConstraint()
        {
            constraints.push_back(new Constraint());
        }

        template <typename Constraint>
        void addConstraint(MeshPtr &mesh, GeomPtr &geom)
        {
            constraints.push_back(new Constraint(mesh, geom));
        }

    private:
        SurfaceEnergy *energy;
        MeshPtr mesh;
        GeomPtr geom;
        Eigen::MatrixXd origPositions;
        unsigned int stepCount;
        std::vector<Constraints::ConstraintBase *> constraints;

        double prevStep;
        void SaveCurrentPositions();
        void RestorePositions();
        void SetGradientStep(Eigen::MatrixXd &gradient, double delta);
    };
} // namespace rsurfaces