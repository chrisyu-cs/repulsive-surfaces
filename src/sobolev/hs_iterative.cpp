#include "sobolev/hs_iterative.h"
#include "sobolev/hs_schur.h"

namespace rsurfaces
{
    namespace Hs
    {
        void ProjectConstrainedHsIterative(Hs::HsMetric &hs, Eigen::VectorXd &gradient, Eigen::VectorXd &dest)
        {
            Eigen::SparseMatrix<double> constraintBlock = hs.GetConstraintBlock(false);
            
            size_t nRows = hs.topLeftNumRows();
            Eigen::VectorXd correction;
            correction.setZero(nRows);

            // Apply A^{-1} to the differential
            ProjectUnconstrainedHsIterative(hs, gradient, dest);

            // Compute the Schur complement part
            UnprojectedSchurCorrection<IterativeInverse>(hs, gradient, correction);
            // Apply A^{-1} from scratch to get the correction
            hs.InvertMetric(correction, correction);

            dest = dest + correction;
        }

    } // namespace Hs
} // namespace rsurfaces
