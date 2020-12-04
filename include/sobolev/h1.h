#pragma once

#include "rsurface_types.h"
#include "matrix_utils.h"
#include "sobolev/constraints.h"

namespace rsurfaces
{
    namespace H1
    {
        void getTriplets(std::vector<Triplet> &triplets, MeshPtr &mesh, GeomPtr &geom, double epsilon, bool premultiplyMass = false);
        void ProjectGradient(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, MeshPtr &mesh, GeomPtr &geom, bool useMass = true);
        void ProjectConstraints(MeshPtr &mesh, GeomPtr &geom, std::vector<ConstraintPack> &constraints,
            std::vector<Constraints::SimpleProjectorConstraint*> simpleConstraints, int newtonIterations);
    }
} // namespace rsurfaces