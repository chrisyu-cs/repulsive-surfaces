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
        Vector3 LeftMultiply(Vector3 v) const
        {
            // Interpret v as a row vector multiplied on the left.
            // Then each entry is just the dot of v with the corresponding column.
            double x = dot(v, col1);
            double y = dot(v, col2);
            double z = dot(v, col3);
            return Vector3{x, y, z};
        }

        // Multiplies J * v.
        Vector3 RightMultiply(Vector3 v) const
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

        void Print() const
        {
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

    Jacobian operator+(const Jacobian &a, const Jacobian &b);
    Jacobian operator-(const Jacobian &a, const Jacobian &b);
    Jacobian operator*(const Jacobian &a, double c);
    Jacobian operator*(double c, const Jacobian &a);

    inline bool findVertexInTriangle(GCFace &face, GCVertex &vert, GCHalfedge &output)
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
            he = he.next();
        } while (he != start);

        output = he;
        return found;
    }

    namespace SurfaceDerivs
    {
        // Even though this should technically be a 3x3 Jacobian, because
        // its only non-zeros are just a constant along the diagonal,
        // we can just treat it as scalar multiplication.
        template <typename Element>
        inline double barycenterWrtVertex(Element &face, GCVertex &wrt)
        {
            return 0;
        }

        template <>
        inline double barycenterWrtVertex(GCFace &face, GCVertex &wrt)
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
            return mass / count;
        }

        template <typename Element>
        inline Jacobian normalWrtVertex(const GeomPtr &geom, Element &face, GCVertex &wrt)
        {
            return Jacobian{Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
        }

        template <>
        inline Jacobian normalWrtVertex(const GeomPtr &geom, GCFace &face, GCVertex &wrt)
        {
            double area = geom->faceAreas[face];
            Vector3 N = geom->faceNormals[face];
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

        template <typename Element>
        inline Vector3 triangleAreaWrtVertex(const GeomPtr &geom, Element &face, GCVertex &wrt)
        {
            return Vector3{0, 0, 0};
        }

        template <>
        inline Vector3 triangleAreaWrtVertex(const GeomPtr &geom, GCFace &face, GCVertex &wrt)
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
            Vector3 N = geom->faceNormals[face];
            return cross(N, u) / 2.0;
        }

        inline Vector3 meanCurvatureNormal(GCVertex v, GeomPtr &geom)
        {
            Vector3 sum{0, 0, 0};
            for (GCFace f : v.adjacentFaces())
            {
                Vector3 aGrad = triangleAreaWrtVertex(geom, f, v);
                sum += aGrad;
            }
            return sum;
        }
    }; // namespace SurfaceDerivs

} // namespace rsurfaces
