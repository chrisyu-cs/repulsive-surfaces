#include "sobolev/hs_schur.h"

namespace rsurfaces
{
    namespace Hs
    {
        void GetSchurComplement(HsMetric &hs, std::vector<ConstraintPack> constraints, SchurComplement &dest)
        {
            size_t nVerts = hs.mesh->nVertices();
            size_t compNRows = 0;
            size_t bigNRows = hs.topLeftNumRows();

            // Figure out how many rows the constraint block is
            for (ConstraintPack &c : constraints)
            {
                compNRows += c.constraint->nRows();
            }
            if (compNRows == 0)
            {
                std::cout << "No constraints provided to Schur complement." << std::endl;
                throw 1;
            }

            dest.C.setZero(compNRows, bigNRows);
            size_t curRow = 0;

            // Fill in the constraint block by getting the entries for each constraint
            // while incrementing the rows
            for (ConstraintPack &c : constraints)
            {
                c.constraint->addEntries(dest.C, hs.mesh, hs.geom, curRow);
                curRow += c.constraint->nRows();
            }

            // https://en.wikipedia.org/wiki/Schur_complement
            // We want to compute (M/A) = D - C A^{-1} B.
            // In our case, D = 0, and B = C^T, so this is C A^{-1} C^T.
            // Unfortunately this means we have to apply A^{-1} once for each column of C^T,
            // which could get expensive if we have too many constraints.

            // First allocate some space for a single column
            Eigen::VectorXd curCol;
            curCol.setZero(bigNRows);
            // And some space for A^{-1} C^T
            Eigen::MatrixXd A_inv_CT;
            A_inv_CT.setZero(bigNRows, compNRows);

            // For each column, copy it into curCol, and do the solve for A^{-1}
            for (size_t r = 0; r < compNRows; r++)
            {
                // Copy the row of C into the column
                for (size_t i = 0; i < 3 * nVerts; i++)
                {
                    curCol(i) = dest.C(r, i);
                }
                hs.InvertMetric(curCol, curCol);
                // Copy the column into the column of A^{-1} C^T
                for (size_t i = 0; i < bigNRows; i++)
                {
                    A_inv_CT(i, r) = curCol(i);
                }
            }

            // Now we've multiplied A^{-1} C^T, so just multiply this with C and negate it
            dest.M_A = -dest.C * A_inv_CT;
        }

        void ProjectViaSchur(HsMetric &hs, Eigen::MatrixXd &gradient, Eigen::MatrixXd &dest, SchurComplement &comp)
        {
            size_t nVerts = hs.mesh->nVertices();

            Eigen::VectorXd temp;
            temp.setZero(comp.C.cols());
            MatrixUtils::MatrixIntoColumn(gradient, temp);
            
            ProjectViaSchurV(hs, temp, temp, comp);

            MatrixUtils::ColumnIntoMatrix(temp, dest);
        }

        void ProjectViaSchurV(HsMetric &hs, Eigen::VectorXd &curCol, Eigen::VectorXd &dest, SchurComplement &comp)
        {
            size_t nVerts = hs.mesh->nVertices();
            // Invert the "saddle matrix" now:
            // the block of M^{-1} we want is A^{-1} + A^{-1} C^T (M/A)^{-1} C A^{-1}
            Eigen::VectorXd tempCol = curCol;
            hs.InvertMetric(tempCol, tempCol);

            // Now we compute the correction.
            // Again we already have A^{-1} once, so no need to recompute it
            Eigen::VectorXd C_Ai_x = comp.C * tempCol;
            Eigen::VectorXd MAi_C_Ai_x;
            MAi_C_Ai_x.setZero(C_Ai_x.rows());
            MatrixUtils::SolveDenseSystem(comp.M_A, C_Ai_x, MAi_C_Ai_x);
            Eigen::VectorXd B_MAi_C_Ai_x = comp.C.transpose() * MAi_C_Ai_x;
            // Apply A^{-1} from scratch one more time
            hs.InvertMetric(B_MAi_C_Ai_x, B_MAi_C_Ai_x);

            dest = tempCol + B_MAi_C_Ai_x;
        }

        void ProjectSchurConstraints(HsMetric &hs, std::vector<ConstraintPack> &constraints, SchurComplement &comp, int newtonSteps)
        {
            size_t nRows = comp.M_A.rows();
            int nIters = 0;
            Eigen::VectorXd vals(nRows);

            while (nIters < newtonSteps)
            {
                vals.setZero();
                int curRow = 0;
                // Fill right-hand side with error values
                for (ConstraintPack &c : constraints)
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
                MatrixUtils::SolveDenseSystem(comp.M_A, vals, vals);
                // Apply C^T
                Eigen::VectorXd correction = comp.C.transpose() * vals;
                // Apply A^{-1}
                hs.InvertMetric(correction, correction);

                // Apply the correction to the vertex positions
                VertexIndices verts = hs.mesh->getVertexIndices();
                size_t nVerts = hs.mesh->nVertices();
                for (GCVertex v : hs.mesh->vertices())
                {
                    int base = 3 * verts[v];
                    Vector3 vertCorr{correction(base), correction(base + 1), correction(base + 2)};
                    hs.geom->inputVertexPositions[v] += vertCorr;
                }

                vals.setZero();
                curRow = 0;
                // Fill right-hand side with error values
                for (ConstraintPack &c : constraints)
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