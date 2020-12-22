#pragma once

#include "hs.h"

namespace rsurfaces
{

    namespace Hs
    {
        void ProjectViaSchurV(const HsMetric &hs, Eigen::VectorXd &gradient, Eigen::VectorXd &dest);
        void ProjectViaSchur(const HsMetric &hs, Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest);
        void ProjectSchurConstraints(const HsMetric &hs, int newtonSteps);
    } // namespace Hs

} // namespace rsurfaces