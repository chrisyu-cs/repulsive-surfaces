#pragma once

#include "rsurface_types.h"
#include "matrix_utils.h"

namespace rsurfaces
{
    namespace Constraints
    {
        class ConstraintBase
        {
        public:
            virtual ~ConstraintBase() {}
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow) = 0;
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow) = 0;
            virtual void addValue(Eigen::VectorXd &V, MeshPtr &mesh, GeomPtr &geom, int baseRow) = 0;
            virtual double getTargetValue() = 0;
            virtual void incrementTargetValue(double incr) = 0;
            virtual size_t nRows() = 0;
        };

        void addEntriesToSymmetric(ConstraintBase &cs, Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
        void addTripletsToSymmetric(ConstraintBase &cs, std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);

        class SimpleProjectorConstraint : public ConstraintBase
        {
        public:
            virtual void ProjectConstraint(MeshPtr &mesh, GeomPtr &geom) = 0;
        };
        
    } // namespace Constraints

    struct ConstraintPack
    {
        Constraints::ConstraintBase *constraint;
        double stepSize;
        long iterationsLeft;
    };

} // namespace rsurfaces