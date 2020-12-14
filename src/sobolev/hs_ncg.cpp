#include "sobolev/hs_ncg.h"

namespace rsurfaces
{
    namespace Hs
    {
        HsNCG::HsNCG(HsMetric *hs_)
        {
            hs = hs_;
            resetFlag = true;
        }

        double HsNCG::computeB_PolakRibiere(Eigen::MatrixXd &l2DiffCurr, Eigen::MatrixXd &hsGradCurr)
        {
            // Recall that the system we solve for the gradient is
            // A * (Hs gradient) = (L2 differential)
            // So (L2)^T (x) = (A * Hs)^T (x) = Hs^T A x, meaning this
            // evaluates the Hs inner product / norm.
            double numer = (l2DiffCurr.transpose() * (hsGradCurr - hsGradPrev)).trace();
            double denom = (l2DiffPrev.transpose() * hsGradPrev).trace();
            return numer / denom;
        }

        void HsNCG::ComputeNCGDirection(Eigen::MatrixXd &l2DiffCurr, Eigen::MatrixXd &hsOutput)
        {
            if (resetFlag)
            {
                // The first conjugate direction s_0 is just the first projected gradient
                conjugateDir.setZero(l2DiffCurr.rows(), l2DiffCurr.cols());
                hs->ProjectViaSparseMat(l2DiffCurr, conjugateDir);
                // Search direction is negative of gradient
                conjugateDir *= -1;

                // Save "previous step" values
                l2DiffPrev = l2DiffCurr;
                hsGradPrev = conjugateDir;
                resetFlag = false;
            }
            else
            {
                conjugateDir.setZero(l2DiffCurr.rows(), l2DiffCurr.cols());
                hs->ProjectViaSparseMat(l2DiffCurr, conjugateDir);

            }
        }

    } // namespace Hs

} // namespace rsurfaces
