#pragma once

#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace Constraints
    {
        class VertexPinConstraint : public ConstraintBase
        {
        public:
            VertexPinConstraint(MeshPtr &mesh, GeomPtr &geom, std::vector<size_t> indices_);
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addValue(Eigen::VectorXd &V, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual double getTargetValue();
            virtual void incrementTargetValue(double incr);
            virtual size_t nRows();
        private:
            std::vector<size_t> indices;
            std::vector<Vector3> initPositions;
        };
    } // namespace Constraints
} // namespace rsurfaces