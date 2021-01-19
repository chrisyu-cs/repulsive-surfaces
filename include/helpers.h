#pragma once

#include "rsurface_types.h"
#include <vector>
#include <unordered_set>
#include <chrono>

namespace rsurfaces
{

    inline GCVertex getOppositeVertex(GCEdge &e, GCVertex &v)
    {
        GCHalfedge he = e.halfedge();
        if (he.vertex() == v)
        {
            return he.twin().vertex();
        }
        else
        {
            return he.vertex();
        }
    }

    // Find the set of vertices that are on the boundary of either
    // triangle, without duplicates
    inline void GetVerticesWithoutDuplicates(GCFace f1, GCFace f2, std::vector<GCVertex> &verts)
    {
        std::unordered_set<GCVertex> vertInds;
        for (GCVertex v : f1.adjacentVertices())
        {
            if (vertInds.count(v) == 0)
            {
                verts.push_back(v);
                vertInds.insert(v);
            }
        }
        for (GCVertex v : f2.adjacentVertices())
        {
            if (vertInds.count(v) == 0)
            {
                verts.push_back(v);
                vertInds.insert(v);
            }
        }
    }

    // Find the set of triangles adjacent to either vertex, without duplicates
    inline void GetFacesWithoutDuplicates(GCVertex v1, GCVertex v2, std::vector<GCFace> &faces)
    {
        std::unordered_set<GCFace> fInds;
        for (GCFace f : v1.adjacentFaces())
        {
            if (fInds.count(f) == 0)
            {
                faces.push_back(f);
                fInds.insert(f);
            }
        }
        for (GCFace f : v2.adjacentFaces())
        {
            if (fInds.count(f) == 0)
            {
                faces.push_back(f);
                fInds.insert(f);
            }
        }
    }


    inline void MultiplyVecByMass(Eigen::VectorXd &A, MeshPtr const &mesh, GeomPtr const &geom)
    {
        VertexIndices inds = mesh->getVertexIndices();
        for (GCVertex v : mesh->vertices())
        {
            int i = inds[v];
            A(3 * i) *= geom->vertexDualAreas[v];
            A(3 * i + 1) *= geom->vertexDualAreas[v];
            A(3 * i + 2) *= geom->vertexDualAreas[v];
        }
    }

    inline void MultiplyByMass(Eigen::MatrixXd &A, MeshPtr const &mesh, GeomPtr const &geom)
    {
        VertexIndices inds = mesh->getVertexIndices();
        for (GCVertex v : mesh->vertices())
        {
            A.row(inds[v]) *= geom->vertexDualAreas[v];
        }
    }

    inline void MultiplyByInvMass(Eigen::MatrixXd &A, MeshPtr const &mesh, GeomPtr const &geom)
    {
        VertexIndices inds = mesh->getVertexIndices();
        for (GCVertex v : mesh->vertices())
        {
            A.row(inds[v]) /= geom->vertexDualAreas[v];
        }
    }

    inline Vector3 GetRow(Eigen::MatrixXd &A, int i)
    {
        return Vector3{A(i, 0), A(i, 1), A(i, 2)};
    }

    inline long currentTimeMilliseconds()
    {
        using namespace std::chrono;
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    }

    inline int sgn_fn(double x)
    {
        if (x > 0)
            return 1;
        else if (x < 0)
            return -1;
        else
            return 0;
    }

    inline void scaleMesh(GeomPtr &geom, MeshPtr &mesh, double scale, Vector3 center)
    {
        for (GCVertex v : mesh->vertices())
        {
            Vector3 disp = geom->inputVertexPositions[v] - center;
            geom->inputVertexPositions[v] = center + scale * disp;
        }
    }

    inline void translateMesh(GeomPtr &geom, MeshPtr &mesh, Vector3 offset)
    {
        for (GCVertex v : mesh->vertices())
        {
            geom->inputVertexPositions[v] += offset;
        }
    }

    inline void translatePoints(GeomPtr &geom, MeshPtr &mesh, Vector3 offset, std::vector<GCVertex> &points)
    {
        for (GCVertex v : points)
        {
            geom->inputVertexPositions[v] += offset;
        }
    }

    inline double MeanCurvature(GCVertex v, const MeshPtr &mesh, const GeomPtr &geom)
    {
        double sum = 0;
        for (GCEdge e : v.adjacentEdges())
        {
            double dih = geom->edgeDihedralAngles[e];
            sum += dih * geom->edgeLength(e);
        }
        return sum / 4;
    }

    template <typename GPtr, typename MPtr>
    inline Vector3 meshBarycenter(GPtr const &geom, MPtr const &mesh)
    {
        Vector3 center{0, 0, 0};
        double sumWeight = 0;
        for (GCVertex v : mesh->vertices())
        {
            center += geom->inputVertexPositions[v] * geom->vertexDualAreas[v];
            sumWeight += geom->vertexDualAreas[v];
        }
        return center / sumWeight;
    }

    template <typename GPtr, typename MPtr>
    inline Vector3 barycenterOfPoints(GPtr const &geom, MPtr const &mesh, std::vector<GCVertex> &points)
    {
        Vector3 center{0, 0, 0};
        double sumWeight = 0;
        for (GCVertex v : points)
        {
            center += geom->inputVertexPositions[v] * geom->vertexDualAreas[v];
            sumWeight += geom->vertexDualAreas[v];
        }
        return center / sumWeight;
    }

    template <typename GPtr, typename MPtr, typename Src>
    inline Vector3 averageOfVectors(GPtr const &geom, MPtr const &mesh, Src const &vec)
    {
        Vector3 sum{0, 0, 0};
        double sumWeight = 0;
        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            Vector3 val = Vector3{vec(3 * i), vec(3 * i + 1), vec(3 * i + 2)};
            GCVertex v = mesh->vertex(i);
            sum += val * geom->vertexDualAreas[v];
            sumWeight += geom->vertexDualAreas[v];
        }
        return sum / sumWeight;
    }

    template <typename GPtr, typename MPtr>
    inline Vector3 averageOfMatrixRows(GPtr const &geom, MPtr const &mesh, Eigen::MatrixXd const &mat)
    {
        Vector3 sum{0, 0, 0};
        double sumWeight = 0;

        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            GCVertex v = mesh->vertex(i);
            Vector3 val = Vector3{mat(i, 0), mat(i, 1), mat(i, 2)};
            sum += val * geom->vertexDualAreas[v];
            sumWeight += geom->vertexDualAreas[v];
        }
        return sum / sumWeight;
    }

    template <typename Dst>
    inline void addShiftToVectors(Dst &vec, size_t nTriples, Vector3 shift)
    {
        for (size_t i = 0; i < nTriples; i++)
        {
            vec(3 * i) += shift.x;
            vec(3 * i + 1) += shift.y;
            vec(3 * i + 2) += shift.z;
        }
    }

    inline void addShiftToMatrixRows(Eigen::MatrixXd &mat, size_t nRows, Vector3 shift)
    {
        for (size_t i = 0; i < nRows; i++)
        {
            mat(i, 0) += shift.x;
            mat(i, 1) += shift.y;
            mat(i, 2) += shift.z;
        }
    }

    inline Vector3 faceNormal(GeomPtr const &geom, GCFace f)
    {
        return geom->faceNormals[f];
    }

    inline Vector3 faceNormal(GeomPtr const &geom, MassNormalPoint f)
    {
        return f.normal;
    }

    inline double faceArea(GeomPtr const &geom, GCFace f)
    {
        return geom->faceAreas[f];
    }

    inline double faceArea(GeomPtr const &geom, MassNormalPoint f)
    {
        return f.mass;
    }

    inline Vector3 vertexAreaNormalUnnormalized(GeomPtr const &geom, GCVertex v)
    {
        Vector3 normal{0, 0, 0};
        for (GCFace f : v.adjacentFaces())
        {
            normal += geom->faceArea(f) * geom->faceNormal(f);
        }
        return normal;
    }

    inline Vector3 vertexAreaNormal(GeomPtr const &geom, GCVertex v)
    {
        Vector3 n = vertexAreaNormalUnnormalized(geom, v);
        return n.normalize();
    }

    inline Vector3 vectorOfHalfedge(GeomPtr const &geom, GCHalfedge he)
    {
        // Vector points from tail to head
        return geom->inputVertexPositions[he.twin().vertex()] - geom->inputVertexPositions[he.vertex()];
    }

    inline Vector3 areaWeightedNormal(GeomPtr const &geom, GCVertex v)
    {
        Vector3 sum{0, 0, 0};
        for (GCFace f : v.adjacentFaces())
        {
            if (f.isBoundaryLoop())
            {
                continue;
            }
            // Get two vectors from the face
            Vector3 e1 = vectorOfHalfedge(geom, f.halfedge());
            Vector3 e2 = vectorOfHalfedge(geom, f.halfedge().next());
            // Area-weighted normal is 1/2 cross product of the two edges
            Vector3 AN = cross(e1, e2) / 2;
            sum += AN;
        }
        return sum;
    }

    inline double totalArea(GeomPtr const &geom, MeshPtr const &mesh)
    {
        double area = 0;
        for (GCFace v : mesh->faces())
        {
            area += geom->faceArea(v);
        }
        return area;
    }

    inline double faceVolume(GeomPtr const &geom, GCFace f)
    {
        Vector3 v1 = geom->inputVertexPositions[f.halfedge().vertex()];
        Vector3 v2 = geom->inputVertexPositions[f.halfedge().next().vertex()];
        Vector3 v3 = geom->inputVertexPositions[f.halfedge().next().next().vertex()];
        return dot(cross(v1, v2), v3) / 6;
    }

    inline double totalVolume(GeomPtr const &geom, MeshPtr const &mesh)
    {
        double area = 0;
        for (GCFace f : mesh->faces())
        {
            area += faceVolume(geom, f);
        }
        return area;
    }

    inline Vector3 faceBarycenter(GeomPtr const &geom, GCFace f)
    {
        Vector3 sum{0, 0, 0};
        int count = 0;
        for (GCVertex v : f.adjacentVertices())
        {
            sum += geom->inputVertexPositions[v];
            count++;
        }
        return sum / count;
    }

    inline Vector3 faceBarycenter(GeomPtr const &geom, MassNormalPoint f)
    {
        return f.point;
    }

    inline Vector3 vectorMin(Vector3 v1, Vector3 v2)
    {
        return Vector3{fmin(v1.x, v2.x), fmin(v1.y, v2.y), fmin(v1.z, v2.z)};
    }

    inline Vector3 vectorMax(Vector3 v1, Vector3 v2)
    {
        return Vector3{fmax(v1.x, v2.x), fmax(v1.y, v2.y), fmax(v1.z, v2.z)};
    }

} // namespace rsurfaces
