#pragma once

#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace Constraints
    {
        class BarycenterConstraint : public ConstraintBase
        {
        public:
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual size_t nRows();
        };

        class BarycenterConstraint3X : public ConstraintBase
        {
        public:
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual size_t nRows();
        };
    } // namespace Constraints
} // namespace rsurfaces