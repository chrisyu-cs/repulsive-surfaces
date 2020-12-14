#pragma once

#include "rsurface_types.h"
#include "sobolev/constraints.h"
#include "surface_energy.h"
#include "line_search.h"

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

        void RecenterMesh();
        void ResetAllConstraints();
        void ResetAllPotentials();
        
        void UpdateEnergies();

        template <typename Constraint>
        Constraint* addSchurConstraint(MeshPtr &mesh, GeomPtr &geom, double multiplier, long iterations)
        {
            Constraint *c = new Constraint(mesh, geom);
            double stepSize = 0;
            if (iterations > 0)
            {
                double initVal = c->getTargetValue();
                double change = multiplier * initVal - initVal;
                stepSize = change / iterations;
            }
            schurConstraints.push_back(ConstraintPack{c, stepSize, iterations});
            return c;
        }

        template <typename Constraint>
        Constraint* addSimpleConstraint(MeshPtr &mesh, GeomPtr &geom)
        {
            Constraint *c = new Constraint(mesh, geom);
            simpleConstraints.push_back(c);
            return c;
        }


    private:
        std::vector<SurfaceEnergy *> energies;
        MeshPtr mesh;
        GeomPtr geom;
        Eigen::MatrixXd origPositions;
        unsigned int stepCount;
        std::vector<ConstraintPack> schurConstraints;
        std::vector<Constraints::SimpleProjectorConstraint*> simpleConstraints;
        Vector3 origBarycenter;

        double prevStep;
    };
} // namespace rsurfaces