#pragma once

#include "implicit/implicit_surface.h"
#include "surface_energy.h"

namespace rsurfaces
{
    class ImplicitAttractor : public SurfaceEnergy
    {
        public:
        ImplicitAttractor(MeshPtr mesh_, GeomPtr geom_, std::unique_ptr<ImplicitSurface> surface_, UVDataPtr uvs_, double power_, double w);

        virtual double Value();
        virtual void Differential(Eigen::MatrixXd &output);
        virtual Vector2 GetExponents();
        virtual OptimizedClusterTree *GetBVH();
        virtual double GetTheta();

        private:
        double power;
        UVDataPtr uvs;
        std::unique_ptr<ImplicitSurface> surface;

        inline bool shouldAttract(GCVertex v)
        {
            // If no UVs are defined, then everything gets attracted
            if (!uvs)
            {
                return true;
            }
            else
            {
                // If there are UVs, then only those with positive (nonzero)
                // x-values get attracted
                for (GCCorner c : v.adjacentCorners())
                {
                    if ((*uvs)[c].x > 1e-10)
                    {
                        return true;
                    }
                }
                // If no UV on this vertex is positive, no attraction
                return false;
            }
        }
    };
}