#pragma once

#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace Constraints
    {
        class VertexPinConstraint : public SimpleProjectorConstraint
        {
        public:
            VertexPinConstraint(MeshPtr &mesh, GeomPtr &geom);
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addErrorValues(Eigen::VectorXd &V, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual size_t nRows();
            virtual void ProjectConstraint(MeshPtr &mesh, GeomPtr &geom);
        
            void pinVertices(MeshPtr &mesh, GeomPtr &geom, std::vector<size_t> &indices_);

        private:
            std::vector<size_t> indices;
            std::vector<Vector3> initPositions;
        };
    } // namespace Constraints
} // namespace rsurfaces