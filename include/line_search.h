#pragma once

#include "rsurface_types.h"
#include "surface_energy.h"

namespace rsurfaces
{
    const double LS_STEP_THRESHOLD = 1e-20;

    inline double GetEnergyValue(std::vector<SurfaceEnergy*> &energies)
    {
        double sum = 0;
        for (SurfaceEnergy *energy : energies)
        {
            sum += energy->Value();
        }
        return sum;
    }

    inline void AddGradientsToMatrix(std::vector<SurfaceEnergy*> &energies, Eigen::MatrixXd &gradient)
    {
        for (SurfaceEnergy *energy : energies)
        {
            energy->Differential(gradient);
        }
    }

    class LineSearch
    {
        public:
        LineSearch(MeshPtr mesh_, GeomPtr geom_, std::vector<SurfaceEnergy*> energies_, double maxStep_=-1.);
        double BacktrackingLineSearch(Eigen::MatrixXd &gradient, double initGuess, double gradDot, bool negativeIsForward = true);
        
        private:
        MeshPtr mesh;
        GeomPtr geom;
        std::vector<SurfaceEnergy*> energies;
        Eigen::MatrixXd origPositions;
        double maxStep;

        void SaveCurrentPositions();
        void RestorePositions();
        void SetGradientStep(Eigen::MatrixXd &gradient, double delta);
    };
}

