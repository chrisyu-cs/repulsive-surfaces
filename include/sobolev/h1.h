#pragma once

#include "rsurface_types.h"
#include "matrix_utils.h"
#include "sobolev/constraints.h"
#include "sobolev/sparse_factorization.h"

namespace rsurfaces
{
    namespace H1
    {
        void getTriplets(std::vector<Triplet> &triplets, const MeshPtr &mesh, const GeomPtr &geom, double epsilon, bool premultiplyMass = false);
        void ProjectGradient(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, MeshPtr &mesh, GeomPtr &geom, bool useMass = true);
        void ProjectConstraints(MeshPtr &mesh, GeomPtr &geom, std::vector<Constraints::SimpleProjectorConstraint *> simpleConstraints,
                                std::vector<ConstraintPack> &newtonConstraints, SparseFactorization &factoredL, int newtonIterations);
    } // namespace H1
} // namespace rsurfaces