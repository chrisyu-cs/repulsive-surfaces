#include "sobolev/l2_lbfgs.h"

namespace rsurfaces
{
    L2_LBFGS::L2_LBFGS(size_t memSize_) : LBFGSOptimizer(memSize_)
    {
    }

    void L2_LBFGS::ApplyInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output)
    {
        output = input;
    }

    void L2_LBFGS::ApplyInverseInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output)
    {
        output = input;
    }

    void L2_LBFGS::SetUpInnerProduct(MeshPtr &mesh, GeomPtr &geom)
    {
        // do nothing
    }
} // namespace rsurfaces