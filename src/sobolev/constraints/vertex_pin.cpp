#include "sobolev/constraints/vertex_pin.h"

namespace rsurfaces
{
    namespace Constraints
    {
        VertexPinConstraint::VertexPinConstraint(const MeshPtr &mesh, const GeomPtr &geom) {}

        void VertexPinConstraint::pinVertices(const MeshPtr &mesh, const GeomPtr &geom, std::vector<size_t> &pinData)
        {
            for (size_t i = 0; i < pinData.size(); i++)
            {
                indices.push_back(pinData[i]);
                initPositions.push_back(geom->inputVertexPositions[mesh->vertex(pinData[i])]);
                offsets.push_back(PinOffset{Vector3{0, 0, 0}, 0});
            }
        }

        void VertexPinConstraint::pinVertices(const MeshPtr &mesh, const GeomPtr &geom, std::vector<VertexPinData> &pinData)
        {
            for (size_t i = 0; i < pinData.size(); i++)
            {
                indices.push_back(pinData[i].vertID);
                initPositions.push_back(geom->inputVertexPositions[mesh->vertex(pinData[i].vertID)]);

                if (pinData[i].iterations == 0)
                {
                    offsets.push_back(PinOffset{Vector3{0, 0, 0}, 0});
                }
                else
                {
                    Vector3 offsetStep = pinData[i].offset / pinData[i].iterations;
                    offsets.push_back(PinOffset{offsetStep, pinData[i].iterations});
                }
            }
        }

        void VertexPinConstraint::ResetFunction(const MeshPtr &mesh, const GeomPtr &geom)
        {
            for (size_t i = 0; i < indices.size(); i++)
            {
                initPositions[i] = geom->inputVertexPositions[mesh->vertex(indices[i])];
            }
        }

        void VertexPinConstraint::addTriplets(std::vector<Triplet> &triplets, const MeshPtr &mesh, const GeomPtr &geom, int baseRow)
        {
            // All we do is put a 1 in the index for all pinned vertices
            for (size_t i = 0; i < indices.size(); i++)
            {
                triplets.push_back(Triplet(baseRow + 3 * i, 3 * indices[i], 1));
                triplets.push_back(Triplet(baseRow + 3 * i + 1, 3 * indices[i] + 1, 1));
                triplets.push_back(Triplet(baseRow + 3 * i + 2, 3 * indices[i] + 2, 1));
            }
        }

        void VertexPinConstraint::addEntries(Eigen::MatrixXd &M, const MeshPtr &mesh, const GeomPtr &geom, int baseRow)
        {
            // All we do is put a 1 in the index for all pinned vertices
            for (size_t i = 0; i < indices.size(); i++)
            {
                M(baseRow + 3 * i, 3 * indices[i]) = 1;
                M(baseRow + 3 * i + 1, 3 * indices[i] + 1) = 1;
                M(baseRow + 3 * i + 2, 3 * indices[i] + 2) = 1;
            }
        }

        void VertexPinConstraint::addErrorValues(Eigen::VectorXd &V, const MeshPtr &mesh, const GeomPtr &geom, int baseRow)
        {
            for (size_t i = 0; i < indices.size(); i++)
            {
                Vector3 current = geom->inputVertexPositions[mesh->vertex(indices[i])];
                V(baseRow + 3 * i) = current.x - initPositions[i].x;
                V(baseRow + 3 * i + 1) = current.y - initPositions[i].y;
                V(baseRow + 3 * i + 2) = current.z - initPositions[i].z;
            }
        }

        size_t VertexPinConstraint::nRows()
        {
            // Every vertex gets 3 rows, one to freeze each of its coordinates
            return indices.size() * 3;
        }

        void VertexPinConstraint::ProjectConstraint(MeshPtr &mesh, GeomPtr &geom)
        {
            for (size_t i = 0; i < indices.size(); i++)
            {
                GCVertex v_i = mesh->vertex(indices[i]);
                geom->inputVertexPositions[v_i] = initPositions[i];

                if (offsets[i].numIterations > 0)
                {
                    initPositions[i] += offsets[i].offsetStep;
                    offsets[i].numIterations--;
                }
            }
        }

    } // namespace Constraints
} // namespace rsurfaces
