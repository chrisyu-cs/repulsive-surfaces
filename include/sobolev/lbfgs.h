#pragma once

#include "rsurface_types.h"
#include <list>

namespace rsurfaces
{
    class LBFGSOptimizer
    {
        public:
        LBFGSOptimizer(size_t memSize_);
        virtual void ApplyInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output);
        virtual void ApplyInverseInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output);
        virtual void SetUpInnerProduct(MeshPtr &mesh, GeomPtr &geom);
        Eigen::VectorXd& direction();
        void UpdateDirection(Eigen::VectorXd &currentPosition, Eigen::VectorXd &currentGradient);
        void ResetMemory();

        protected:
        size_t memSize;
        bool firstStep;
        std::list<Eigen::VectorXd> s_list;
        std::list<Eigen::VectorXd> y_list;
        std::vector<double> rhos;
        std::vector<double> alphas;
        Eigen::VectorXd z;

        Eigen::VectorXd lastPosition;
        Eigen::VectorXd lastGradient;
    };
}