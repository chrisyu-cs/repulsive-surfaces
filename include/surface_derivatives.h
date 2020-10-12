#pragma once

#include "rsurface_types.h"
#include <iomanip>
#include "helpers.h"
#include "matrix_utils.h"

namespace rsurfaces
{
    struct Jacobian
    {
        Vector3 col1;
        Vector3 col2;
        Vector3 col3;

        // Multiplies v^T * J.
        // Left-multiplying takes in a desired change in the (vector-valued) function,
        // and outputs what spatial direction to move in to achieve that change.
        inline Vector3 LeftMultiply(Vector3 v) const
        {
            // Interpret v as a row vector multiplied on the left.
            // Then each entry is just the dot of v with the corresponding column.
            double x = dot(v, col1);
            double y = dot(v, col2);
            double z = dot(v, col3);
            return Vector3{x, y, z};
        }

        // Multiplies J * v.
        // Right-multiplying takes a spatial direction, and outputs the change
        // in the function value that would result from moving that way.
        inline Vector3 RightMultiply(Vector3 v) const
        {
            // Interpret v as a column vector multiplied on the right.
            double x = col1.x * v.x + col2.x * v.y + col3.x * v.z;
            double y = col1.y * v.x + col2.y * v.y + col3.y * v.z;
            double z = col1.z * v.x + col2.z * v.y + col3.z * v.z;
            return Vector3{x, y, z};
        }

        inline void SetFromMatrix3(Eigen::Matrix3d &M)
        {
            col1 = Vector3{M(0, 0), M(1, 0), M(2, 0)};
            col2 = Vector3{M(0, 1), M(1, 1), M(2, 1)};
            col3 = Vector3{M(0, 2), M(1, 2), M(2, 2)};
        }

        // Add the entries of J^T to the matrix in a 3x3 block starting from
        // the two coordinates.
        // For constraint matrices, we want J^T because this produces spatial
        // directions, which we ultimately want to constrain.
        inline void AddTransposeToMatrix(Eigen::MatrixXd &M, size_t topLeftRow, size_t topLeftCol)
        {
            M(topLeftRow + 0, topLeftCol + 0) += col1.x;
            M(topLeftRow + 0, topLeftCol + 1) += col1.y;
            M(topLeftRow + 0, topLeftCol + 2) += col1.z;

            M(topLeftRow + 1, topLeftCol + 0) += col2.x;
            M(topLeftRow + 1, topLeftCol + 1) += col2.y;
            M(topLeftRow + 1, topLeftCol + 2) += col2.z;

            M(topLeftRow + 2, topLeftCol + 0) += col3.x;
            M(topLeftRow + 2, topLeftCol + 1) += col3.y;
            M(topLeftRow + 2, topLeftCol + 2) += col3.z;
        }

        // Same as above, but adding sparse triplets instead.
        // For constraint matrices, we want J^T because this produces spatial
        // directions, which we ultimately want to constrain.
        inline void AddTransposeTriplets(std::vector<Triplet> &triplets, size_t topLeftRow, size_t topLeftCol)
        {
            triplets.push_back(Triplet(topLeftRow + 0, topLeftCol + 0, col1.x));
            triplets.push_back(Triplet(topLeftRow + 0, topLeftCol + 1, col1.y));
            triplets.push_back(Triplet(topLeftRow + 0, topLeftCol + 2, col1.z));

            triplets.push_back(Triplet(topLeftRow + 1, topLeftCol + 0, col2.x));
            triplets.push_back(Triplet(topLeftRow + 1, topLeftCol + 1, col2.y));
            triplets.push_back(Triplet(topLeftRow + 1, topLeftCol + 2, col2.z));
            
            triplets.push_back(Triplet(topLeftRow + 2, topLeftCol + 0, col3.x));
            triplets.push_back(Triplet(topLeftRow + 2, topLeftCol + 1, col3.y));
            triplets.push_back(Triplet(topLeftRow + 2, topLeftCol + 2, col3.z));
        }

        inline void Print() const
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
    Jacobian operator-(const Jacobian &a);
    Jacobian operator*(const Jacobian &a, double c);
    Jacobian operator*(double c, const Jacobian &a);
    Jacobian operator/(const Jacobian &a, double c);

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
                if (f.isBoundaryLoop())
                {
                    continue;
                }
                Vector3 aGrad = triangleAreaWrtVertex(geom, f, v);
                sum += aGrad;
            }
            return sum;
        }

        inline Jacobian triAreaNormalWrtVertex(const GeomPtr &geom, GCFace &f, GCVertex &wrt)
        {
            double area = geom->faceAreas[f];
            Vector3 normal = geom->faceNormals[f];

            Vector3 dArea = triangleAreaWrtVertex(geom, f, wrt);
            Jacobian dNormal = normalWrtVertex(geom, f, wrt);

            Jacobian s1 = Jacobian::OuterProductToJacobian(normal, dArea);
            Jacobian s2 = area * dNormal;

            Jacobian sum = s1 + s2;
            return sum;
        }

        inline Jacobian vertexNormalUWrtVertex(const GeomPtr &geom, GCVertex &vert, GCVertex &wrt)
        {
            Jacobian J{Vector3{0, 0, 0}, Vector3{0, 0, 0}, Vector3{0, 0, 0}};
            if (vert == wrt)
            {
                // If differentiating by same vertex, then differentiate
                // (area * normal) for all surrounding faces
                for (GCFace f : vert.adjacentFaces())
                {
                    if (f.isBoundaryLoop())
                    {
                        continue;
                    }
                    J = J + triAreaNormalWrtVertex(geom, f, wrt);
                }
                return J;
            }

            else
            {
                // Otherwise, need to differentiate (area * normal) for
                // the two shared faces
                GCHalfedge connecting;
                bool found = false;
                // Find the half-edge connecting the two
                for (GCHalfedge he : wrt.incomingHalfedges())
                {
                    if (he.vertex() == vert)
                    {
                        connecting = he;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    return J;
                }
                // Need to differentiate (area * normal) on both sides
                for (GCFace f : connecting.edge().adjacentFaces())
                {
                    if (f.isBoundaryLoop())
                    {
                        continue;
                    }
                    J = J + triAreaNormalWrtVertex(geom, f, wrt);
                }
                return J;
            }
        }

        inline Jacobian vertexNormalWrtVertex(const GeomPtr &geom, GCVertex &vert, GCVertex &wrt)
        {
            // Normal times area
            Vector3 AN = vertexAreaNormalUnnormalized(geom, vert);
            // Derivative of AN
            Jacobian dAN = vertexNormalUWrtVertex(geom, vert, wrt);
            // We want to differentiate N = AN / |AN|, so we'll use the quotient rule:
            // dN = (AN * d|AN| - dAN * |AN|) / |AN|^2
            double norm_AN = AN.norm();
            // Chain rule: d|AN| = (AN / |AN|) * dAN
            Vector3 d_norm_AN = dAN.LeftMultiply(AN / norm_AN);

            Jacobian dN = (Jacobian::OuterProductToJacobian(AN, d_norm_AN) - dAN * norm_AN) / (norm_AN * norm_AN);
            return -dN;
        }
    }; // namespace SurfaceDerivs

} // namespace rsurfaces
