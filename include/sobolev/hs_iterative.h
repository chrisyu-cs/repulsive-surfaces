#pragma once

#include "sobolev/hs.h"
#include "bct_matrix_replacement.h"

namespace rsurfaces
{
    namespace Hs
    {
        void ProjectHsGradientIterative(Hs::HsMetric &hs, Eigen::VectorXd &gradient, Eigen::VectorXd &dest);
    } // namespace Hs
} // namespace rsurfaces
