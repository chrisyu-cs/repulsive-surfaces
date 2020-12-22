#pragma once

#include "sobolev/hs.h"
#include "bct_matrix_replacement.h"

namespace rsurfaces
{
    namespace Hs
    {
        void ProjectUnconstrainedHsIterative(Hs::HsMetric &hs, Eigen::VectorXd &gradient, Eigen::VectorXd &dest, bool includeNewton = false);
    } // namespace Hs
} // namespace rsurfaces
