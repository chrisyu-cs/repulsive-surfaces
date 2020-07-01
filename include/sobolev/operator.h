#pragma once

#include <Eigen/Core>
#include "rsurface_types.h"

namespace rsurfaces {

    class SurfaceOperator {
        virtual void Assemble(MeshPtr mesh, GeomPtr geom) = 0;
        virtual void Solve(Eigen::VectorXd &b, Eigen::VectorXd &output) = 0;
    };

}
