#include "sobolev/hs_ncg.h"

namespace rsurfaces
{
    namespace Hs
    {
        HsNCG::HsNCG(SurfaceEnergy *energy_, std::vector<Constraints::SimpleProjectorConstraint *> &spcs_)
            : spcs(spcs_)
        {
            energy = energy_;
            resetFlag = true;
        }

        double HsNCG::computeB_PolakRibiere(Eigen::MatrixXd &l2Diff_n, Eigen::MatrixXd &delta_xn)
        {
            // Recall that the system we solve for the gradient is
            // A * (Hs gradient) = (L2 differential)
            // So (L2)^T (x) = (A * Hs)^T (x) = Hs^T A x, meaning this
            // evaluates the Hs inner product / norm.
            double numer = (l2Diff_n.transpose() * (delta_xn - delta_xn_1)).trace();
            double denom = (l2Diff_n_1.transpose() * delta_xn_1).trace();

            std::cout << numer << " / " << denom << std::endl;
            return numer / denom;
        }

        double HsNCG::UpdateConjugateDir(Eigen::MatrixXd &l2Diff_n, Hs::HsMetric &hs)
        {
            if (resetFlag)
            {
                // The first conjugate direction s_0 is just the first projected gradient
                conjugateDir.setZero(l2Diff_n.rows(), l2Diff_n.cols());
                // hs.InvertMetricMat(l2Diff_n, conjugateDir);
                Eigen::PartialPivLU<Eigen::MatrixXd> solver;
                hs.ProjectGradientExact(l2Diff_n, conjugateDir, solver);
                double hsGradNorm = (l2Diff_n.transpose() * conjugateDir).trace();
                std::cout << "l2Diff_n norm = " << l2Diff_n.norm() << std::endl;
                std::cout << "delta_xn norm = " << conjugateDir.norm() << std::endl;
                std::cout << "hsGradNorm = " << hsGradNorm << std::endl;
                // Search direction is negative of gradient
                conjugateDir *= -1;

                // Save "previous step" values
                l2Diff_n_1 = l2Diff_n;
                delta_xn_1 = conjugateDir;
                resetFlag = false;

                return hsGradNorm;
            }
            else
            {
                // Step 1: compute Hs gradient direction
                Eigen::MatrixXd delta_xn;
                delta_xn.setZero(l2Diff_n.rows(), l2Diff_n.cols());
                // hs.InvertMetricMat(l2Diff_n, delta_xn);
                Eigen::PartialPivLU<Eigen::MatrixXd> solver;
                hs.ProjectGradientExact(l2Diff_n, delta_xn, solver);
                double hsGradNorm = (l2Diff_n.transpose() * delta_xn).trace();
                std::cout << "l2Diff_n norm = " << l2Diff_n.norm() << std::endl;
                std::cout << "delta_xn norm = " << delta_xn.norm() << std::endl;
                std::cout << "hsGradNorm = " << hsGradNorm << std::endl;
                // Search direction is negative of gradient
                delta_xn *= -1;

                // Step 2: compute beta
                double beta = computeB_PolakRibiere(l2Diff_n, delta_xn);

                std::cout << "Computed beta = " << beta << std::endl;
                beta = fmax(beta, 0);

                // Step 3: update conjugate direction
                conjugateDir = delta_xn + beta * conjugateDir;

                // Save "previous step" values
                l2Diff_n_1 = l2Diff_n;
                delta_xn_1 = delta_xn;

                return hsGradNorm;
            }
        }

        Eigen::MatrixXd &HsNCG::direction()
        {
            return conjugateDir;
        }
    } // namespace Hs

} // namespace rsurfaces
