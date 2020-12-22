#pragma once

#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace Constraints
    {
        class VertexPinConstraint : public SimpleProjectorConstraint
        {
        public:
            VertexPinConstraint(const MeshPtr &mesh, const GeomPtr &geom);
            virtual void ResetFunction(const MeshPtr &mesh, const GeomPtr &geom);
            virtual void addTriplets(std::vector<Triplet> &triplets, const MeshPtr &mesh, const GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, const MeshPtr &mesh, const GeomPtr &geom, int baseRow);
            virtual void addErrorValues(Eigen::VectorXd &V, const MeshPtr &mesh, const GeomPtr &geom, int baseRow);
            virtual size_t nRows();
            virtual void ProjectConstraint(MeshPtr &mesh, GeomPtr &geom);
        
            void pinVertices(const MeshPtr &mesh, const GeomPtr &geom, std::vector<size_t> &indices_);

        private:
            std::vector<size_t> indices;
            std::vector<Vector3> initPositions;
        };
    } // namespace Constraints
} // namespace rsurfaces