#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace Constraints
    {
        void addBarycenterTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            geom->requireVertexDualAreas();
            VertexIndices indices = mesh->getVertexIndices();
            double sumArea = 0;
            for (GCVertex v : mesh->vertices())
            {
                sumArea += geom->vertexDualAreas[v];
            }

            for (GCVertex v : mesh->vertices())
            {
                double wt = geom->vertexDualAreas[v] / sumArea;
                triplets.push_back(Triplet(baseRow, indices[v], wt));
                triplets.push_back(Triplet(indices[v], baseRow, wt));
            }
        }

        void addBarycenterEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            geom->requireVertexDualAreas();
            VertexIndices indices = mesh->getVertexIndices();
            double sumArea = 0;
            for (GCVertex v : mesh->vertices())
            {
                sumArea += geom->vertexDualAreas[v];
            }

            for (GCVertex v : mesh->vertices())
            {
                double wt = geom->vertexDualAreas[v] / sumArea;
                M(baseRow, indices[v]) = wt;
                M(indices[v], baseRow) = wt;
            }
        }

        Vector3 getMeshBarycenter(MeshPtr &mesh, GeomPtr &geom)
        {
            double totalMass = 0;
            Vector3 avg{0, 0, 0};

            for (GCVertex v : mesh->vertices())
            {
                totalMass += geom->vertexDualAreas[v];
                avg += geom->inputVertexPositions[v] * geom->vertexDualAreas[v];
            }
            avg /= totalMass;
            return avg;
        }

        void addScalingTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {

            VertexIndices indices = mesh->getVertexIndices();
            for (GCVertex v : mesh->vertices())
            {
                Vector3 outward = geom->inputVertexPositions[v];
                int colBase = 3 * indices[v];
                triplets.push_back(Triplet(baseRow, colBase, outward.x));
                triplets.push_back(Triplet(baseRow + 1, colBase + 1, outward.y));
                triplets.push_back(Triplet(baseRow + 2, colBase + 2, outward.z));
            }
        }

        inline double MeanCurvature(GCVertex v, MeshPtr &mesh, GeomPtr &geom)
        {
            double sum = 0;
            for (GCEdge e : v.adjacentEdges())
            {
                double dih = geom->edgeDihedralAngles[e];
                sum += dih * geom->edgeLength(e);
            }
            return sum / 4;
        }

        void addScalingEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            geom->requireEdgeDihedralAngles();
            Vector3 center = getMeshBarycenter(mesh, geom);

            VertexIndices indices = mesh->getVertexIndices();
            for (GCVertex v : mesh->vertices())
            {
                // Vector3 outward = geom->inputVertexPositions[v] - center;
                Vector3 outward = geom->vertexNormals[v] * MeanCurvature(v, mesh, geom);

                int colBase = 3 * indices[v];
                M(baseRow, colBase) = outward.x;
                M(baseRow + 1, colBase + 1) = outward.y;
                M(baseRow + 2, colBase + 2) = outward.z;

                M(colBase, baseRow) = outward.x;
                M(colBase + 1, baseRow + 1) = outward.y;
                M(colBase + 2, baseRow + 2) = outward.z;
            }
            geom->unrequireEdgeDihedralAngles();
        }

    } // namespace Constraints
} // namespace rsurfaces