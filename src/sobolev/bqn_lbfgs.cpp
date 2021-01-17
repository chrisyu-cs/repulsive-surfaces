#include "sobolev/h1.h"
#include "sobolev/bqn_lbfgs.h"

namespace rsurfaces
{
    BQN_LBFGS::BQN_LBFGS(size_t memSize_, std::vector<Constraints::SimpleProjectorConstraint *> simpleConstraints_, double bqn_B_)
    : H1_LBFGS(memSize_, simpleConstraints_)
    {
        bqn_B = bqn_B_;
    }

    void BQN_LBFGS::UpdateHistory(Eigen::VectorXd &currentPosition, Eigen::VectorXd &currentGradient)
    {
        // Update memory vectors based on current position and gradient
        Eigen::VectorXd y_current = currentGradient - lastGradient;
        Eigen::VectorXd s_current = currentPosition - lastPosition;
        // Difference in positions (secant difference) gets updated normally
        s_list.push_back(s_current);

        // Difference in gradients gets blended with Laplacian difference in positions Ls_i
        Eigen::VectorXd Ls_current;
        Ls_current.setZero(s_current.rows());
        ApplyInnerProduct(s_current, Ls_current);
        // beta_i = proj[0,1] (normest(L) y_t^T L s_i) / B(T)
        // (equation 13 of BCQN / Zhu et al)
        double normL = L.norm();
        double beta_i = (normL * y_current.dot(Ls_current)) / bqn_B;
        beta_i = fmax(fmin(beta_i, 1), 0);

        // Blend between y_i and Ls_i
        Ls_current = (1 - beta_i) * y_current + beta_i * Ls_current;
        y_list.push_back(Ls_current);

        std::cout << "  * Did blended update with beta_i = " << beta_i << std::endl;

        // Evict oldest vectors if at size limit
        if (s_list.size() > memSize)
        {
            s_list.pop_front();
            y_list.pop_front();
        }
        lastGradient = currentGradient;
        lastPosition = currentPosition;
    }


}
