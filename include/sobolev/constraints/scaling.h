#pragma once

#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace Constraints
    {
        class ScalingConstraint : public SaddleMatrixConstraint
        {
        public:
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addValue(Eigen::VectorXd &V, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual double getTargetValue();
            virtual void incrementTargetValue(double incr);
            virtual size_t nRows();
        };
    } // namespace Constraints
} // namespace rsurfaces