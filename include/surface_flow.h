#pragma once

#include <memory>

#include "rsurface_types.h"
#include "sobolev/constraints.h"
#include "sobolev/all_constraints.h"
#include "surface_energy.h"
#include "line_search.h"
#include "sobolev/hs_ncg.h"
#include "sobolev/lbfgs.h"

namespace rsurfaces
{
    class SurfaceFlow
    {
    public:
        SurfaceFlow(SurfaceEnergy *energy_);
        void AddAdditionalEnergy(SurfaceEnergy *extraEnergy);

        void StepL2Unconstrained();
        void StepL2Projected();
        void StepProjectedGradientExact();
        void StepProjectedGradient();
        void StepProjectedGradientIterative();
        void StepH1LBFGS();
        void StepBQN();

        void StepH1ProjGrad();
        void StepAQP(double invKappa);

        SurfaceEnergy *BaseEnergy();

        void RecenterMesh();
        void ResetAllConstraints();
        void ResetAllPotentials();

        void AssembleGradients(Eigen::MatrixXd &dest);
        std::unique_ptr<Hs::HsMetric> GetHsMetric();

        void UpdateEnergies();
        double evaluateEnergy();

        template <typename Constraint>
        Constraint *addSchurConstraint(MeshPtr &mesh, GeomPtr &geom, double multiplier, long iterations)
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
        Constraint *addSimpleConstraint(MeshPtr &mesh, GeomPtr &geom)
        {
            Constraint *c = new Constraint(mesh, geom);
            simpleConstraints.push_back(c);
            return c;
        }

        bool allowBarycenterShift;
        bool verticesMutated;
        bool disableNearField;

    private:
        std::vector<SurfaceEnergy *> energies;
        MeshPtr mesh;
        GeomPtr geom;
        Eigen::MatrixXd prevPositions1;
        Eigen::MatrixXd prevPositions2;
        unsigned int stepCount;
        std::vector<ConstraintPack> schurConstraints;
        std::vector<Constraints::SimpleProjectorConstraint *> simpleConstraints;
        Vector3 origBarycenter;
        Hs::HsNCG *ncg;
        Constraints::BarycenterComponentsConstraint *secretBarycenter;
        LBFGSOptimizer* lbfgs;

        size_t addConstraintTriplets(std::vector<Triplet> &triplets, bool includeSchur);
        
        void prefactorConstrainedLaplacian(SparseFactorization &factored, bool includeSchur);
        void prefactorConstrainedLaplacian(Eigen::SparseMatrix<double> &L, SparseFactorization &factored, bool includeSchur);

        inline void incrementSchurConstraints()
        {
            for (ConstraintPack &c : schurConstraints)
            {
                if (c.iterationsLeft > 0)
                {
                    c.iterationsLeft--;
                    c.constraint->incrementTargetValue(c.stepSize);
                }
            }
        }

        double bqn_B;

        double prevStep;
    };
} // namespace rsurfaces