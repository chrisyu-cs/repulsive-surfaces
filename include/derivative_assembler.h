#pragma once

#include "rsurface_types.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>


namespace rsurfaces
{
    Eigen::SparseMatrix<double> DerivativeAssembler(MeshPtr const & mesh, GeomPtr const & geom);
} // namespace rsurfaces
