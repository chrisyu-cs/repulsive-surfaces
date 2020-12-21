#pragma once

#include "sobolev/hs.h"

namespace rsurfaces
{
    namespace Hs
    {
        class HsIterative
        {
        public:
            HsIterative(SurfaceEnergy *energy_, std::vector<Constraints::SimpleProjectorConstraint *> &spcs_);
            void ProjectGradientIterative(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, std::vector<ConstraintPack> &schurConstraints);

        private:
            SurfaceEnergy *energy;
            std::vector<Constraints::SimpleProjectorConstraint *> &spcs;
        };
    } // namespace Hs
} // namespace rsurfaces
