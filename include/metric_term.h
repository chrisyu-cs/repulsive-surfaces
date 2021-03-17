#pragma once

#include "rsurface_types.h"
#include "optimized_bct.h"

namespace rsurfaces
{
    class MetricTerm
    {
        public:
        virtual ~MetricTerm() {}
        virtual void MultiplyAdd(Eigen::VectorXd &vec, Eigen::VectorXd &result) const = 0;
    };

    class BCTMetricTerm : public MetricTerm
    {
        public:
        BCTMetricTerm(std::shared_ptr<OptimizedBlockClusterTree> bct_)
        {
            bct = bct_;
        }
        
        virtual void MultiplyAdd(Eigen::VectorXd &vec, Eigen::VectorXd &result) const
        {
            bct->MultiplyV3(vec, result, BCTKernelType::HighOrder, true);
            if (!bct->disableNearField)
            {
                bct->MultiplyV3(vec, result, rsurfaces::BCTKernelType::LowOrder, true);
            }
        }

        private:
        std::shared_ptr<OptimizedBlockClusterTree> bct;
    };
}

