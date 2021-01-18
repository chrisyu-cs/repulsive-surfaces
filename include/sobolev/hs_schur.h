#pragma once

#include "hs.h"

namespace rsurfaces
{

    namespace Hs
    {
        template <typename Inverse>
        void ProjectViaSchurV(const HsMetric &hs, Eigen::VectorXd &curCol, Eigen::VectorXd &dest);

        template <typename Inverse>
        void ProjectViaSchur(const HsMetric &hs, Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest);

        template <typename Inverse>
        void ProjectSchurConstraints(const HsMetric &hs, int newtonSteps);

        template <typename Inverse>
        void UnprojectedSchurCorrection(const HsMetric &hs, Eigen::VectorXd hsGradient, Eigen::VectorXd &dest)
        {
            // After the end of this function, need to apply A^{-1} C^T to dest
        }

        template <typename Inverse>
        void ProjectViaSchurV(const HsMetric &hs, Eigen::VectorXd &curCol, Eigen::VectorXd &dest)
        {
            size_t nVerts = hs.mesh->nVertices();
            // Invert the "saddle matrix" now:
            // the block of M^{-1} we want is A^{-1} + A^{-1} C^T (M/A)^{-1} C A^{-1}
            Eigen::VectorXd Ainv_g = curCol;
            std::cout << "  Applying metric inverse for gradient..." << std::endl;
            Inverse::Apply(hs, Ainv_g, Ainv_g);

            // Now we compute the correction
            // Start from hsGradient = A^{-1} x
            Eigen::VectorXd C_Ai_x = hs.Schur<Inverse>().C * Ainv_g;
            Eigen::VectorXd MAi_C_Ai_x;
            MAi_C_Ai_x.setZero(C_Ai_x.rows());
            MatrixUtils::SolveDenseSystem(hs.Schur<Inverse>().M_A, C_Ai_x, MAi_C_Ai_x);
            // Use the cached A^{-1} C^T to do this last step
            std::cout << "  Applying metric inverse for Schur complement orthogonalization..." << std::endl;
            dest = Ainv_g + hs.Schur<Inverse>().Ainv_CT * MAi_C_Ai_x;
        }

        template <typename Inverse>
        void ProjectViaSchur(const HsMetric &hs, Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest)
        {
            size_t nVerts = hs.mesh->nVertices();

            if (hs.newtonConstraints.size() > 0)
            {
                Eigen::VectorXd temp;
                temp.setZero(hs.Schur<Inverse>().C.cols());
                MatrixUtils::MatrixIntoColumn(gradient, temp);

                ProjectViaSchurV<Inverse>(hs, temp, temp);

                MatrixUtils::ColumnIntoMatrix(temp, dest);
            }
            else
            {
                hs.InvertMetricMat(gradient, dest);
            }
        }

        template <typename Inverse>
        void ProjectSchurConstraints(const HsMetric &hs, int newtonSteps)
        {
            size_t nRows = hs.Schur<Inverse>().M_A.rows();
            int nIters = 0;
            Eigen::VectorXd vals(nRows);

            while (nIters < newtonSteps)
            {
                vals.setZero();
                int curRow = 0;
                // Fill right-hand side with error values
                for (const ConstraintPack &c : hs.newtonConstraints)
                {
                    c.constraint->addErrorValues(vals, hs.mesh, hs.geom, curRow);
                    curRow += c.constraint->nRows();
                }

                double constraintError = vals.lpNorm<Eigen::Infinity>();
                std::cout << "  * Constraint error after " << nIters << " iterations = " << constraintError << std::endl;
                if (nIters > 0 && constraintError < 1e-2)
                {
                    break;
                }

                nIters++;

                // In this case we want the block of the inverse that multiplies the bottom block
                // -A^{-1} B (M/A)^{-1}, where B = C^T
                // Apply (M/A) inverse first
                MatrixUtils::SolveDenseSystem(hs.Schur<Inverse>().M_A, vals, vals);
                // Apply cached A{^-1} C^T
                Eigen::VectorXd correction = hs.Schur<Inverse>().Ainv_CT * vals;

                // Apply the correction to the vertex positions
                VertexIndices verts = hs.mesh->getVertexIndices();
                for (GCVertex v : hs.mesh->vertices())
                {
                    int base = 3 * verts[v];
                    Vector3 vertCorr{correction(base), correction(base + 1), correction(base + 2)};
                    hs.geom->inputVertexPositions[v] += vertCorr;
                }

                vals.setZero();
                curRow = 0;
                // Fill right-hand side with error values
                for (const ConstraintPack &c : hs.newtonConstraints)
                {
                    c.constraint->addErrorValues(vals, hs.mesh, hs.geom, curRow);
                    curRow += c.constraint->nRows();
                }
                double corrError = vals.lpNorm<Eigen::Infinity>();
                std::cout << "  * Corrected error " << constraintError << " -> " << corrError << std::endl;
            }
        }
    } // namespace Hs

} // namespace rsurfaces