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
            virtual void addValue(Eigen::VectorXd &V, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual double getTargetValue();
            virtual void incrementTargetValue(double incr);
            virtual size_t nRows();
        };

        class BarycenterConstraint3X : public ConstraintBase
        {
        public:
            BarycenterConstraint3X(MeshPtr &mesh, GeomPtr &geom);
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual void addValue(Eigen::VectorXd &V, MeshPtr &mesh, GeomPtr &geom, int baseRow);
            virtual double getTargetValue();
            virtual void incrementTargetValue(double incr);
            virtual size_t nRows();

        private:
            Vector3 initValue;
        };
    } // namespace Constraints
} // namespace rsurfaces