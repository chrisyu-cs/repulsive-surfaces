#pragma once

#include "hs.h"

namespace rsurfaces
{

    namespace Hs
    {
        void GetSchurComplement(HsMetric &hs, std::vector<ConstraintPack> constraints, SchurComplement &dest);
        void ProjectViaSchur(HsMetric &hs, Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, SchurComplement &comp);
        void ProjectSchurConstraints(HsMetric &hs, std::vector<ConstraintPack> &constraints, SchurComplement &comp, int newtonSteps);

    } // namespace Hs

} // namespace rsurfaces