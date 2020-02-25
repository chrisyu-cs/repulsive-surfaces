#include "surface_derivatives.h"

namespace rsurfaces
{

Jacobian SurfaceDerivs::barycenterWrtVertex(GCFace &face, GCVertex &wrt)
{
    GCHalfedge start = face.halfedge();
    GCHalfedge he = start;
    double count = 0;
    double mass = 0;
    do
    {
        count += 1;
        if (he.vertex() == wrt)
        {
            mass += 1;
        }
        he = he.next();
    } while (he != start);

    double weight = mass / count;
    return Jacobian{Vector3{weight, 0, 0}, Vector3{0, weight, 0}, Vector3{0, 0, weight}};
}

Jacobian SurfaceDerivs::normalWrtVertex(const GeomPtr &geom, GCFace &face, GCVertex &wrt)
{
    double area = geom->faceArea(face);
    Vector3 N = geom->faceNormal(face);
    // First go to the halfedge that is based at wrt
    GCHalfedge he;
    bool found = findVertexInTriangle(face, wrt, he);
    if (!found)
    {
        return Jacobian{Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
    }
    // The next halfedge is on the opposite edge from wrt
    he = he.next();
    Vector3 e = geom->inputVertexPositions[he.twin().vertex()] - geom->inputVertexPositions[he.vertex()];
    Vector3 exN = 1.0 / (2 * area) * cross(e, N);
    return Jacobian::OuterProductToJacobian(exN, N);
}

Vector3 SurfaceDerivs::triangleAreaWrtVertex(const GeomPtr &geom, GCFace &face, GCVertex &wrt)
{
    // First go to the halfedge that is based at wrt
    GCHalfedge he;
    bool found = findVertexInTriangle(face, wrt, he);
    if (!found)
    {
        return Vector3{0, 0, 0};
    }
    // The next halfedge is on the opposite edge from wrt
    he = he.next();
    Vector3 u = geom->inputVertexPositions[he.twin().vertex()] - geom->inputVertexPositions[he.vertex()];
    // Get the vector pointing away from the opposite edge
    Vector3 N = geom->faceNormal(face);
    return cross(N, u) / 2.0;
}

Jacobian operator+(const Jacobian &a, const Jacobian &b)
{
    Vector3 col1 = a.col1 + b.col1;
    Vector3 col2 = a.col2 + b.col2;
    Vector3 col3 = a.col3 + b.col3;

    return Jacobian{col1, col2, col3};
}

Jacobian operator-(const Jacobian &a, const Jacobian &b)
{
    return a + (b * -1);
}

Jacobian operator*(const Jacobian &a, double c)
{
    return Jacobian{a.col1 * c, a.col2 * c, a.col3 * c};
}

Jacobian operator*(double c, const Jacobian &a)
{
    return a * c;
}

} // namespace rsurfaces