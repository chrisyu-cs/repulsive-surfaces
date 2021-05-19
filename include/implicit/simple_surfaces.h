#pragma once

#include "implicit/implicit_surface.h"

namespace rsurfaces
{
    class ImplicitSphere : public ImplicitSurface {
        public:
        ImplicitSphere(double r, Vector3 c);
        virtual double SignedDistance(Vector3 point);
        virtual Vector3 GradientOfDistance(Vector3 point);
        virtual double BoundingDiameter();
        virtual Vector3 BoundingCenter();

        private:
        double radius;
        Vector3 center;
    };

    // infinite cylinder of radius r centered at c along axis u
    class ImplicitCylinder : public ImplicitSurface {
        public:
        ImplicitCylinder(double r, Vector3 c, Vector3 u);
        virtual double SignedDistance(Vector3 point);
        virtual Vector3 GradientOfDistance(Vector3 point);
        virtual double BoundingDiameter();
        virtual Vector3 BoundingCenter();

        private:
        double radius;
        Vector3 center;
        Vector3 axis;
    };

    class ImplicitTorus : public ImplicitSurface {
        public:
        ImplicitTorus(double major, double minor, Vector3 c);
        virtual double SignedDistance(Vector3 point);
        virtual Vector3 GradientOfDistance(Vector3 point);
        virtual double BoundingDiameter();
        virtual Vector3 BoundingCenter();

        private:
        double majorRadius;
        double minorRadius;
        Vector3 center;
    };

    class FlatPlane : public ImplicitSurface {
        public:
        FlatPlane(Vector3 p, Vector3 n);
        virtual double SignedDistance(Vector3 point2);
        virtual Vector3 GradientOfDistance(Vector3 point2);
        virtual double BoundingDiameter();
        virtual Vector3 BoundingCenter();

        private:
        Vector3 point;
        Vector3 normal;
    };
}
