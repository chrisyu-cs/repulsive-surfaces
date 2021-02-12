#include "implicit/simple_surfaces.h"

namespace rsurfaces
{
    // ========== Implicit sphere of variable radius ==========

    ImplicitSphere::ImplicitSphere(double r, Vector3 c) {
        radius = r;
        center = c;
    }

    double ImplicitSphere::SignedDistance(Vector3 point) {
        return (point - center).norm() - radius;
    }

    Vector3 ImplicitSphere::GradientOfDistance(Vector3 point) {
        return (point - center).normalize();
    }

    double ImplicitSphere::BoundingDiameter() {
        return radius * 2;
    }

    Vector3 ImplicitSphere::BoundingCenter() {
        return center;
    }
    // ============================================================

    // ========== Implicit torus of variable radii ==========

    ImplicitTorus::ImplicitTorus(double major, double minor, Vector3 c) {
        majorRadius = major;
        minorRadius = minor;
        center = c;
    }

    double ImplicitTorus::SignedDistance(Vector3 point) {
        // Vector3 disp = point - center;
        // Vector2 planar{disp.x, disp.z};
        // Vector2 q{norm(planar) - majorRadius, disp.y};
        // return norm(q) - minorRadius;

        Vector3 p = point - center;
        double normAll = norm2(p);
        double normXZ = norm(Vector2{p.x, p.z});
        return sqrt(normAll - 2 * majorRadius * normXZ + majorRadius * majorRadius) - minorRadius;
    }

    Vector3 ImplicitTorus::GradientOfDistance(Vector3 point) {
        Vector3 p = point - center;
        double normAll = norm2(p);
        double normXZ = norm(Vector2{p.x, p.z});
        double sqTerm = sqrt(normAll - 2 * majorRadius * normXZ + majorRadius * majorRadius);

        double partialX = 2 * p.x - (2 * p.x * majorRadius) / normXZ;
        double partialY = 2 * p.y;
        double partialZ = 2 * p.z - (2 * p.z * majorRadius) / normXZ;

        return (1.0 / (2 * sqTerm)) * Vector3{partialX, partialY, partialZ};
    }

    double ImplicitTorus::BoundingDiameter() {
        return majorRadius * 2 + minorRadius * 2;
    }

    Vector3 ImplicitTorus::BoundingCenter() {
        return center;
    }
    // ============================================================

    // ========== Infinite flat plane with given normal ==========
    
    FlatPlane::FlatPlane(Vector3 p, Vector3 n)
    {
        point = p;
        normal = n.normalize();
    }

    double FlatPlane::SignedDistance(Vector3 point2) {
        Vector3 fromPt = point2 - point;
        return dot(normal, fromPt);
    }

    Vector3 FlatPlane::GradientOfDistance(Vector3 point2) {
        return normal;
    }

    double FlatPlane::BoundingDiameter() {
        return 2;
    }

    Vector3 FlatPlane::BoundingCenter() {
        return Vector3{0, 0, 0};
    }
    // ============================================================
}