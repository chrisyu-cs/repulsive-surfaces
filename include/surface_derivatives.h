#pragma once

#include "rsurface_types.h"
#include <iomanip>

namespace rsurfaces
{
struct Jacobian
{
    Vector3 col1;
    Vector3 col2;
    Vector3 col3;

    // Multiplies v^T * J.
    Vector3 LeftMultiply(Vector3 v)
    {
        // Interpret v as a row vector multiplied on the left.
        // Then each entry is just the dot of v with the corresponding column.
        double x = dot(v, col1);
        double y = dot(v, col2);
        double z = dot(v, col3);
        return Vector3{x, y, z};
    }

    // Multiplies J * v.
    Vector3 RightMultiply(Vector3 v)
    {
        // Interpret v as a column vector multiplied on the right.
        double x = col1.x * v.x + col2.x * v.y + col3.x * v.z;
        double y = col1.y * v.x + col2.y * v.y + col3.y * v.z;
        double z = col1.z * v.x + col2.z * v.y + col3.z * v.z;
        return Vector3{x, y, z};
    }

    void SetFromMatrix3(Eigen::Matrix3d M)
    {
        col1 = Vector3{M(0, 0), M(1, 0), M(2, 0)};
        col2 = Vector3{M(0, 1), M(1, 1), M(2, 1)};
        col3 = Vector3{M(0, 2), M(1, 2), M(2, 2)};
    }

    void Print() {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << col1.x << "\t" << col2.x << "\t" << col3.x << std::endl;
        std::cout << col1.y << "\t" << col2.y << "\t" << col3.y << std::endl;
        std::cout << col1.z << "\t" << col2.z << "\t" << col3.z << std::endl;
    }

    static Jacobian OuterProductToJacobian(Vector3 v1, Vector3 v2)
    {
        Vector3 col1 = v1 * v2.x;
        Vector3 col2 = v1 * v2.y;
        Vector3 col3 = v1 * v2.z;
        return Jacobian{col1, col2, col3};
    }
};

Jacobian operator+(const Jacobian& a, const Jacobian& b);
Jacobian operator-(const Jacobian& a, const Jacobian& b);
Jacobian operator*(const Jacobian& a, double c);
Jacobian operator*(double c, const Jacobian& a);

class SurfaceDerivs
{
public:
    static Jacobian barycenterWrtVertex(GCFace &face, GCVertex &wrt);
    static Jacobian normalWrtVertex(const GeomPtr &geom, GCFace &face, GCVertex &wrt);
    static Vector3 triangleAreaWrtVertex(const GeomPtr &geom, GCFace &face, GCVertex &wrt);
};
} // namespace rsurfaces
