#pragma once

#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace Constraints
    {
        class BarycenterConstraint3X : public SimpleProjectorConstraint
        {
        public:
            BarycenterConstraint3X(const MeshPtr &mesh, const GeomPtr &geom);
            virtual void ResetFunction(const MeshPtr &mesh, const GeomPtr &geom);
            virtual void addTriplets(std::vector<Triplet> &triplets, const MeshPtr &mesh, const GeomPtr &geom, int baseRow);
            virtual void addEntries(Eigen::MatrixXd &M, const MeshPtr &mesh, const GeomPtr &geom, int baseRow);
            virtual void addErrorValues(Eigen::VectorXd &V, const MeshPtr &mesh, const GeomPtr &geom, int baseRow);
            virtual size_t nRows();
            virtual void ProjectConstraint(MeshPtr &mesh, GeomPtr &geom);

        private:
            Vector3 initValue;
        };
    } // namespace Constraints
} // namespace rsurfaces