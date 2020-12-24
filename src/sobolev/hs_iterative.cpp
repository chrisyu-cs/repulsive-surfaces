#include "sobolev/hs_iterative.h"
#include "sobolev/hs_schur.h"

namespace rsurfaces
{
    namespace Hs
    {
        void ProjectConstrainedHsIterative(Hs::HsMetric &hs, Eigen::VectorXd &gradient, Eigen::VectorXd &dest)
        {
            if (hs.newtonConstraints.size() > 0)
            {
                ProjectViaSchurV<IterativeInverse>(hs, gradient, dest);
            }
            else
            {
                ProjectUnconstrainedHsIterative(hs, gradient, dest);
            }
        }

        void ProjectConstrainedHsIterativeMat(Hs::HsMetric &hs, Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest)
        {
            Eigen::VectorXd col;
            col.setZero(hs.topLeftNumRows());
            MatrixUtils::MatrixIntoColumn(gradient, col);

            ProjectConstrainedHsIterative(hs, col, col);

            MatrixUtils::ColumnIntoMatrix(col, dest);
        }

    } // namespace Hs
} // namespace rsurfaces
