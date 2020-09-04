#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
    namespace Hs
    {

        inline double MetricDistanceTerm(double s, Vector3 v1, Vector3 v2)
        {
            double dist_term = 1.0 / pow(norm(v1 - v2), 2 * (s - 1) + 2);
            return dist_term;
        }

        inline double MetricDistanceTermFrac(double s, Vector3 v1, Vector3 v2)
        {
            double dist_term = 1.0 / pow(norm(v1 - v2), 2 * s + 2);
            return dist_term;
        }

        template <typename V, typename VF>
        void ApplyMidOperator(const MeshPtr &mesh, const GeomPtr &geom, V &a, VF &out)
        {
            FaceIndices fInds = mesh->getFaceIndices();
            VertexIndices vInds = mesh->getVertexIndices();

            for (GCFace face : mesh->faces())
            {
                double total = 0;
                // Get one-third the value on all adjacent vertices
                for (GCVertex vert : face.adjacentVertices())
                {
                    total += a(vInds[vert]) / 3.0;
                }
                out(fInds[face]) += total;
            }
        }

        template <typename V, typename VF>
        void ApplyMidOperatorTranspose(const MeshPtr &mesh, const GeomPtr &geom, VF &a, V &out)
        {

            FaceIndices fInds = mesh->getFaceIndices();
            VertexIndices vInds = mesh->getVertexIndices();

            for (GCVertex vert : mesh->vertices())
            {
                double total = 0;
                // Put weight = 1/3 on all adjacent faces
                for (GCFace face : vert.adjacentFaces())
                {
                    total += a(fInds[face]) / 3.0;
                }
                out(vInds[vert]) += total;
            }
        }
    } // namespace Hs
} // namespace rsurfaces