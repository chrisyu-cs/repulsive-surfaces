#pragma once

#include "rsurface_types.h"
#include "lbfgs.h"
#include "sobolev/sparse_factorization.h"
#include "sobolev/constraints.h"
#include "sobolev/h1_lbfgs.h"

namespace rsurfaces
{
    class BQN_LBFGS : public H1_LBFGS
    {
        public:
        BQN_LBFGS(size_t memSize_, std::vector<Constraints::SimpleProjectorConstraint *> simpleConstraints_, double bqn_B_);
        virtual void UpdateHistory(Eigen::VectorXd &currentPosition, Eigen::VectorXd &currentGradient);

        private:
        double bqn_B;

    };
}