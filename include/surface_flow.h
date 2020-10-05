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
        void addConstraint(MeshPtr &mesh, GeomPtr &geom, double multiplier, long iterations)
        {
            Constraints::ConstraintBase *c = new Constraint(mesh, geom);
            double stepSize = 0;
            if (iterations > 0)
            {
                double initVal = c->getTargetValue();
                double change = multiplier * initVal - initVal;
                stepSize = change / iterations;
            }
            schurConstraints.push_back(ConstraintPack{c, stepSize, iterations});
        }

    private:
        void UpdateEnergies();
        double GetEnergyValue();
        void AddGradientToMatrix(Eigen::MatrixXd &gradient);

        std::vector<SurfaceEnergy *> energies;
        MeshPtr mesh;
        GeomPtr geom;
        Eigen::MatrixXd origPositions;
        unsigned int stepCount;
        std::vector<ConstraintPack> schurConstraints;
        std::vector<Constraints::SimpleProjectorConstraint> simpleConstraints;
        Vector3 origBarycenter;

        double prevStep;
        void SaveCurrentPositions();
        void RestorePositions();
        void SetGradientStep(Eigen::MatrixXd &gradient, double delta);
    };
} // namespace rsurfaces