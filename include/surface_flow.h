#pragma once

#include "rsurface_types.h"
#include "sobolev/constraints.h"
#include "surface_energy.h"

namespace rsurfaces
{

    class SurfaceFlow
    {
    public:
        SurfaceFlow(SurfaceEnergy *energy_);
        void AddAdditionalEnergy(SurfaceEnergy *extraEnergy);

        void StepNaive(double t);
        void StepFractionalSobolev();
        SurfaceEnergy *BaseEnergy();
        double LineSearchStep(Eigen::MatrixXd &gradient, double initGuess, double gradDot);

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
        void UpdateEnergies();
        double GetEnergyValue();
        void AddGradientToMatrix(Eigen::MatrixXd &gradient);

        std::vector<SurfaceEnergy*> energies;
        MeshPtr mesh;
        GeomPtr geom;
        Eigen::MatrixXd origPositions;
        unsigned int stepCount;
        std::vector<Constraints::ConstraintBase *> constraints;
        Vector3 origBarycenter;

        double prevStep;
        void SaveCurrentPositions();
        void RestorePositions();
        void SetGradientStep(Eigen::MatrixXd &gradient, double delta);
    };
} // namespace rsurfaces