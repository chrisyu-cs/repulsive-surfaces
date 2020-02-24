#include "surface_derivatives.h"

namespace rsurfaces
{

inline bool findVertex(GCFace &face, GCVertex &vert, GCHalfedge &output)
{
    GCHalfedge he = face.halfedge();
    GCHalfedge start = he;
    bool found = false;
    do
    {
        if (he.vertex() == vert)
        {
            found = true;
            break;
        }
        else
            he = he.next();
    } while (he != start);

    output = he;
    return found;
}

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
    bool found = findVertex(face, wrt, he);
    if (!found)
    {
        return Jacobian{Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
    }
    // The next halfedge is on the opposite edge from wrt
    he = he.next();
    Vector3 e = geom->vertexPositions[he.twin().vertex()] - geom->vertexPositions[he.vertex()];
    Vector3 exN = 1.0 / (2 * area) * cross(e, N);
    return Jacobian::OuterProductToJacobian(exN, N);
}

void SurfaceDerivs::numericalCheck(const MeshPtr &mesh, const GeomPtr &geom, double eps)
{
    std::cout << "Did it" << std::endl;
    for (GCFace face : mesh->faces())
    {
        for (GCVertex vert : face.adjacentVertices())
        {
            std::cout << face.getIndex() << " / " << vert.getIndex() << std::endl;
            Jacobian deriv = normalWrtVertex(geom, face, vert);

            Vector3 orig = geom->inputVertexPositions[vert];
            Vector3 origN = geom->faceNormal(face);

            geom->inputVertexPositions[vert] = orig + Vector3{eps, 0, 0};
            Vector3 xN = geom->faceNormal(face);
            Vector3 deriv_x = (xN - origN) / eps;

            geom->inputVertexPositions[vert] = orig + Vector3{0, eps, 0};
            Vector3 yN = geom->faceNormal(face);
            Vector3 deriv_y = (yN - origN) / eps;

            geom->inputVertexPositions[vert] = orig + Vector3{0, 0, eps};
            Vector3 zN = geom->faceNormal(face);
            Vector3 deriv_z = (zN - origN) / eps;

            geom->inputVertexPositions[vert] = orig;

            Jacobian numDeriv{deriv_x, deriv_y, deriv_z};

            std::cout << "Derivative: " << std::endl;
            deriv.Print();
            std::cout << "Numerical derivative: " << std::endl;
            numDeriv.Print();
        }
    }
}

Vector3 SurfaceDerivs::triangleAreaWrtVertex(const GeomPtr &geom, GCFace &face, GCVertex &wrt)
{
    // First go to the halfedge that is based at wrt
    GCHalfedge he;
    bool found = findVertex(face, wrt, he);
    if (!found)
    {
        return Vector3{0, 0, 0};
    }
    // The next halfedge is on the opposite edge from wrt
    he = he.next();
    Vector3 u = geom->vertexPositions[he.twin().vertex()] - geom->vertexPositions[he.vertex()];
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