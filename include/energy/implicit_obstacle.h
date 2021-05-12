#pragma once

#include "implicit/implicit_surface.h"
#include "surface_energy.h"

namespace rsurfaces
{
    class ImplicitObstacle : public SurfaceEnergy
    {
        public:
        ImplicitObstacle(MeshPtr mesh_, GeomPtr geom_, std::unique_ptr<ImplicitSurface> surface_, double power_, double w);

        virtual double Value();
        virtual void Differential(Eigen::MatrixXd &output);
        virtual Vector2 GetExponents();
        virtual OptimizedClusterTree *GetBVH();
        virtual double GetTheta();

        private:
        double power;
        std::unique_ptr<ImplicitSurface> surface;
    };
}