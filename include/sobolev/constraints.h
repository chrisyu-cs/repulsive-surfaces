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
            virtual void addTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow) = 0;
            virtual void addEntries(Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow) = 0;
            virtual void addValue(Eigen::VectorXd &V, MeshPtr &mesh, GeomPtr &geom, int baseRow) = 0;
            virtual size_t nRows() = 0;
        };

        void addEntriesToSymmetric(ConstraintBase &cs, Eigen::MatrixXd &M, MeshPtr &mesh, GeomPtr &geom, int baseRow);
        void addTripletsToSymmetric(ConstraintBase &cs, std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, int baseRow);
    } // namespace Constraints
} // namespace rsurfaces