#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace Constraints
    {
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

        void BarycenterConstraint::addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
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
            }
        }

        void BarycenterConstraint::addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
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
            }
        }

        size_t BarycenterConstraint::nRows()
        {
            return 1;
        }

        void BarycenterConstraint3X::addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            BarycenterConstraint single;
            std::vector<Triplet> singleTriplets;
            single.addTriplets(singleTriplets, mesh, geom, 0);

            for (Triplet t : singleTriplets)
            {
                triplets.push_back(Triplet(baseRow + 3 * t.row(), 3 * t.col(), t.value()));
                triplets.push_back(Triplet(baseRow + 3 * t.row() + 1, 3 * t.col() + 1, t.value()));
                triplets.push_back(Triplet(baseRow + 3 * t.row() + 2, 3 * t.col() + 2, t.value()));
            }
        }

        void BarycenterConstraint3X::addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            Eigen::MatrixXd M_small(1, M.cols());
        }

        size_t BarycenterConstraint3X::nRows()
        {
            return 3;
        }

        void ScalingConstraint::addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow)
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

        void ScalingConstraint::addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow)
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

        size_t ScalingConstraint::nRows()
        {
            return 1;
        }

    } // namespace Constraints
} // namespace rsurfaces