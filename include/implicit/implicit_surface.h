#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
    class ImplicitSurface {
        public:
        virtual ~ImplicitSurface();
        virtual double SignedDistance(Vector3 point) = 0;
        virtual Vector3 GradientOfDistance(Vector3 point) = 0;
        virtual double BoundingDiameter() = 0;
        virtual Vector3 BoundingCenter() = 0;
    };
}