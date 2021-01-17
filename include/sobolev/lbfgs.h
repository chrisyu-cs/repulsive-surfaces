#pragma once

#include "rsurface_types.h"
#include <list>

namespace rsurfaces
{
    class LBFGSOptimizer
    {
        public:
        LBFGSOptimizer(size_t memSize_);
        virtual void ApplyInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output) = 0;
        virtual void ApplyInverseInnerProduct(Eigen::VectorXd &input, Eigen::VectorXd &output) = 0;
        virtual void SetUpInnerProduct(MeshPtr &mesh, GeomPtr &geom) = 0;
        Eigen::VectorXd& direction();
        
        virtual void UpdateHistory(Eigen::VectorXd &currentPosition, Eigen::VectorXd &currentGradient);
        void UpdateDirection(Eigen::VectorXd &currentPosition, Eigen::VectorXd &currentGradient);
        void ResetMemory();

        inline Eigen::VectorXd& y_current()
        {
            return *y_list.rbegin();
        }

        inline Eigen::VectorXd& s_current()
        {
            return *s_list.rbegin();
        }

        inline bool hasHistory()
        {
            return (s_list.size() > 0);
        }


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