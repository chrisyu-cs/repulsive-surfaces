#pragma once

#include "sobolev/hs.h"

namespace rsurfaces {

    using Constraints::ConstraintBase;

    namespace Hs {

        class HsNCG
        {
            public:
            HsNCG(HsMetric *hs_);
            ~HsNCG() {}
            
            void ComputeNCGDirection(Eigen::MatrixXd &l2DiffCurr, Eigen::MatrixXd &hsOutput);

            private:
            HsMetric *hs;
            Eigen::MatrixXd l2DiffPrev;
            Eigen::MatrixXd hsGradPrev;
            Eigen::MatrixXd conjugateDir;
            bool resetFlag;
            
            double computeB_PolakRibiere(Eigen::MatrixXd &l2DiffCurr, Eigen::MatrixXd &hsGradCurr);
        };

    }

}
