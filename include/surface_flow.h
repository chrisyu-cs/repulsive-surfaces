#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{

enum class Preserve {
    Nothing,
    Area,
    Volume
};

class SurfaceFlow
{
public:
    SurfaceFlow(SurfaceEnergy *energy_);
    void StepNaive(double t);
    void StepFractionalSobolev(Preserve what);
    SurfaceEnergy *BaseEnergy();
    double LineSearchStep(Eigen::MatrixXd &gradient, double initGuess, double gradDot);

    void RescaleToPreserveArea(double area);
    void RescaleToPreserveVolume(double area);
    void RecenterMesh();
    const double LS_STEP_THRESHOLD = 1e-20;

private:
    SurfaceEnergy *energy;
    MeshPtr mesh;
    GeomPtr geom;
    Eigen::MatrixXd origPositions;
    unsigned int stepCount;

    double prevStep;
    void SaveCurrentPositions();
    void RestorePositions();
    void SetGradientStep(Eigen::MatrixXd &gradient, double delta);
};
} // namespace rsurfaces