#include "sobolev/constraints/barycenter_components.h"
#include "helpers.h"

#include <deque>

namespace rsurfaces
{
    namespace Constraints
    {
        void fillComponents(const MeshPtr mesh, const GeomPtr geom, std::vector<std::vector<GCVertex>> &components)
        {
            surface::VertexData<bool> grouped(*mesh, false);
            VertexIndices componentIndices(*mesh);

            std::deque<GCVertex> frontier;

            size_t componentID = 0;

            // Run Dijkstra's from each successive ungrouped point
            for (GCVertex vert : mesh->vertices())
            {
                if (grouped[vert])
                {
                    continue;
                }
                std::cout << "Found a new connected component (assigned label " << componentID << ")" << std::endl;

                // Create data for a new connected component
                components.push_back(std::vector<GCVertex>());

                frontier.clear();
                frontier.push_back(vert);

                // Process next and enqueue neighbors
                while (!frontier.empty())
                {
                    GCVertex next = frontier.front();
                    frontier.pop_front();

                    if (grouped[next])
                    {
                        continue;
                    }

                    // Mark this as having been assigned to a connected component
                    grouped[next] = true;
                    components[componentID].push_back(next);

                    for (GCVertex neighbor : next.adjacentVertices())
                    {
                        if (!grouped[neighbor])
                        {
                            frontier.push_back(neighbor);
                        }
                    }
                }

                componentID++;
            }
        }

        void addSingleComponentTriplets(std::vector<Triplet> &triplets, const MeshPtr &mesh, const GeomPtr &geom, std::vector<GCVertex> &comp, int baseRow)
        {
            // Just want to place normalized dual weights in the entry for each vertex
            geom->requireVertexDualAreas();
            VertexIndices indices = mesh->getVertexIndices();
            double sumArea = 0;

            for (GCVertex v : comp)
            {
                sumArea += geom->vertexDualAreas[v];
            }

            for (GCVertex v : comp)
            {
                double wt = geom->vertexDualAreas[v] / sumArea;
                triplets.push_back(Triplet(baseRow, indices[v], wt));
            }
        }

        BarycenterComponentsConstraint::BarycenterComponentsConstraint(const MeshPtr &mesh, const GeomPtr &geom)
        {
            fillComponents(mesh, geom, components);
            ResetFunction(mesh, geom);
        }

        void BarycenterComponentsConstraint::ResetFunction(const MeshPtr &mesh, const GeomPtr &geom)
        {
            componentValues.clear();
            for (std::vector<GCVertex> &comp : components)
            {
                Vector3 center = barycenterOfPoints(geom, mesh, comp);
                componentValues.push_back(center);
            }
        }

        void BarycenterComponentsConstraint::addTriplets(std::vector<Triplet> &triplets, const MeshPtr &mesh, const GeomPtr &geom, int baseRow)
        {
            // Take the same weights from the non-3X version of this constraint,
            // and duplicate them 3 times on each 3x3 diagonal block.
            std::vector<Triplet> singleTriplets;

            for (size_t i = 0; i < components.size(); i++)
            {
                addSingleComponentTriplets(singleTriplets, mesh, geom, components[i], i);
            }

            for (Triplet t : singleTriplets)
            {
                triplets.push_back(Triplet(baseRow + 3 * t.row(), 3 * t.col(), t.value()));
                triplets.push_back(Triplet(baseRow + 3 * t.row() + 1, 3 * t.col() + 1, t.value()));
                triplets.push_back(Triplet(baseRow + 3 * t.row() + 2, 3 * t.col() + 2, t.value()));
            }
        }

        void BarycenterComponentsConstraint::addEntries(Eigen::MatrixXd &M, const MeshPtr &mesh, const GeomPtr &geom, int baseRow)
        {
            std::vector<Triplet> singleTriplets;

            for (size_t i = 0; i < components.size(); i++)
            {
                addSingleComponentTriplets(singleTriplets, mesh, geom, components[i], i);
            }

            for (Triplet t : singleTriplets)
            {
                M(baseRow + 3 * t.row(), 3 * t.col()) = t.value();
                M(baseRow + 3 * t.row() + 1, 3 * t.col() + 1) = t.value();
                M(baseRow + 3 * t.row() + 2, 3 * t.col() + 2) = t.value();
            }
        }

        void BarycenterComponentsConstraint::addErrorValues(Eigen::VectorXd &V, const MeshPtr &mesh, const GeomPtr &geom, int baseRow)
        {
            std::vector<Triplet> singleTriplets;

            for (size_t i = 0; i < components.size(); i++)
            {
                Vector3 current = barycenterOfPoints(geom, mesh, components[i]);
                V(baseRow + 3 * i) = current.x - componentValues[i].x;
                V(baseRow + 3 * i + 1) = current.y - componentValues[i].y;
                V(baseRow + 3 * i + 2) = current.z - componentValues[i].z;
            }
        }

        size_t BarycenterComponentsConstraint::nRows()
        {
            return 3 * components.size();
        }

        void BarycenterComponentsConstraint::ProjectConstraint(MeshPtr &mesh, GeomPtr &geom)
        {
            for (size_t i = 0; i < components.size(); i++)
            {
                Vector3 center = barycenterOfPoints(geom, mesh, components[i]);
                translatePoints(geom, mesh, componentValues[i] - center, components[i]);
            }
        }

    } // namespace Constraints
} // namespace rsurfaces