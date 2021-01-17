#pragma once

#include "rsurface_types.h"
#include "sobolev/lbfgs.h"

namespace rsurfaces
{
    class L2_LBFGS : public LBFGSOptimizer
    {
        public:
        L2_LBFGS(size_t memSize_);
        virtual void ApplyInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output);
        virtual void ApplyInverseInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output);
        virtual void SetUpInnerProduct(MeshPtr &mesh, GeomPtr &geom);
    };
}
