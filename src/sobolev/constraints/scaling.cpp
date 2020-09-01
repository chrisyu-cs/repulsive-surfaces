#include "sobolev/constraints/scaling.h"
#include "helpers.h"

namespace rsurfaces
{
    namespace Constraints
    {
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
            Vector3 center = meshBarycenter(geom, mesh);

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

        void ScalingConstraint::addValue(Eigen::VectorXd &V, MeshPtr &mesh, GeomPtr &geom, int baseRow)
        {
            std::cerr << "Backprojecting scale not supported" << std::endl;
            throw 1;
        }

        size_t ScalingConstraint::nRows()
        {
            return 1;
        }
    } // namespace Constraints
} // namespace rsurfaces