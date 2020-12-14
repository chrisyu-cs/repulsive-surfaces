#pragma once

#include "sobolev/hs.h"

namespace rsurfaces
{

    using Constraints::ConstraintBase;

    namespace Hs
    {

        class HsNCG
        {
        public:
            HsNCG(SurfaceEnergy *energy_, std::vector<Constraints::SimpleProjectorConstraint *> &spcs_);
            ~HsNCG() {}

            double UpdateConjugateDir(Eigen::MatrixXd &l2DiffCurr);
            Eigen::MatrixXd &direction();

            inline void ResetMemory()
            {
                resetFlag = true;
            }

        private:
            Eigen::MatrixXd l2Diff_n_1;
            Eigen::MatrixXd delta_xn_1;
            Eigen::MatrixXd conjugateDir;
            bool resetFlag;
            SurfaceEnergy *energy;
            std::vector<Constraints::SimpleProjectorConstraint *> &spcs;

            double computeB_PolakRibiere(Eigen::MatrixXd &l2DiffCurr, Eigen::MatrixXd &hsGradCurr);
        };

    } // namespace Hs

} // namespace rsurfaces
