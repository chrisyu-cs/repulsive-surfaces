#include "sobolev/lbfgs.h"

namespace rsurfaces
{
    LBFGSOptimizer::LBFGSOptimizer(size_t memSize_)
    {
        memSize = memSize_;
        firstStep = true;
    }

    Eigen::VectorXd &LBFGSOptimizer::direction()
    {
        return z;
    }

    void LBFGSOptimizer::ResetMemory()
    {
        s_list.clear();
        y_list.clear();
        firstStep = true;
    }

    void LBFGSOptimizer::UpdateHistory(Eigen::VectorXd &currentPosition, Eigen::VectorXd &currentGradient)
    {
        // Update memory vectors based on current position and gradient
        Eigen::VectorXd y_current = currentGradient - lastGradient;
        Eigen::VectorXd s_current = currentPosition - lastPosition;
        s_list.push_back(s_current);
        y_list.push_back(y_current);

        // Evict oldest vectors if at size limit
        if (s_list.size() > memSize)
        {
            s_list.pop_front();
            y_list.pop_front();
        }
        lastGradient = currentGradient;
        lastPosition = currentPosition;
    }

    void LBFGSOptimizer::UpdateDirection(Eigen::VectorXd &currentPosition, Eigen::VectorXd &currentGradient)
    {
        if (!firstStep)
        {
            UpdateHistory(currentPosition, currentGradient);
        }
        else
        {
            firstStep = false;
            lastGradient = currentGradient;
            lastPosition = currentPosition;
            z.setZero(lastGradient.rows());
            ApplyInverseInnerProduct(lastGradient, z);
            std::cout << "  * Using just H1 inverse" << std::endl;
            return;
        }

        rhos.resize(s_list.size());
        alphas.resize(s_list.size());

        Eigen::VectorXd q = currentGradient;
        Eigen::VectorXd temp;
        temp.setZero(currentGradient.rows());

        // Compute rho values
        auto s_i_for = s_list.begin();
        auto y_i_for = y_list.begin();
        size_t i = 0;

        while (s_i_for != s_list.end())
        {
            ApplyInnerProduct(*s_i_for, temp);
            double rho = 1.0 / ((*y_i_for).dot(temp));
            rhos[i] = rho;
            s_i_for++;
            y_i_for++;
            i++;
        }

        // Iterate backwards to compute alphas
        auto s_i_rev = s_list.rbegin();
        auto y_i_rev = y_list.rbegin();
        i = s_list.size() - 1;

        while (s_i_rev != s_list.rend())
        {
            ApplyInnerProduct(q, temp);
            double alpha = rhos[i] * ((*s_i_rev).dot(temp));
            alphas[i] = alpha;
            q = q - alphas[i] * (*y_i_rev);

            s_i_rev++;
            y_i_rev++;
            i--;
        }

        // Compute gamma_k = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1}
        ApplyInnerProduct(*(y_list.rbegin()), temp);
        double numer_k = (*(s_list.rbegin())).dot(temp);
        double denom_k = (*(y_list.rbegin())).dot(temp);
        double gamma_k = numer_k / denom_k;

        // Compute initial guess for z
        ApplyInverseInnerProduct(q, z);
        z *= gamma_k;

        // Iterate forward to compute betas
        s_i_for = s_list.begin();
        y_i_for = y_list.begin();
        i = 0;

        while (s_i_for != s_list.end())
        {
            ApplyInnerProduct(z, temp);
            double beta_i = rhos[i] * (*y_i_for).dot(temp);
            z = z + (*s_i_for) * (alphas[i] - beta_i);

            s_i_for++;
            y_i_for++;
            i++;
        }
    }

} // namespace rsurfaces
