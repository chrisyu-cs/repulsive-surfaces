#include "sobolev/hs_iterative.h"

namespace rsurfaces
{
    namespace Hs
    {
        HsIterative::HsIterative(SurfaceEnergy *energy_, std::vector<Constraints::SimpleProjectorConstraint *> &spcs_)
            : spcs(spcs_)
        {
            energy = energy_;
        }

        void HsIterative::ProjectGradientIterative(Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, std::vector<ConstraintPack> &schurConstraints)
        {
            std::cout << "TODO" << std::endl;
        }

    } // namespace Hs
} // namespace rsurfaces
