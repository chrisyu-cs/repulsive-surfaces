#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
    enum class BCTKernelType
    {
        FractionalOnly,
        HighOrder,
        LowOrder
    };

    // Multiplies C * v and C^T * lambda as though these were the constraint
    // rows of a saddle matrix, and adds the result to b.
    template <typename V, typename Dest, typename Mat>
    void MultiplyConstraintBlock(const MeshPtr &mesh, const V &v, Dest &b, Mat &C, bool addToResult)
    {
        size_t nConstraints = C.rows();
        size_t nV3 = 3 * mesh->nVertices();

        // Assume that C is (nConstraints x nV3)
        if (addToResult)
        {
            // C * v goes into the bottom block of b
            b.block(nV3, 0, nConstraints, 1) += C * v.block(0, 0, nV3, 1);
            // C^T * lambda goes into the top block of b
            b.block(0, 0, nV3, 1) += C.transpose() * v.block(nV3, 0, nConstraints, 1);
        }
        else
        {
            // C * v goes into the bottom block of b
            b.block(nV3, 0, nConstraints, 1) = C * v.block(0, 0, nV3, 1);
            // C^T * lambda goes into the top block of b
            b.block(0, 0, nV3, 1) = C.transpose() * v.block(nV3, 0, nConstraints, 1);
        }
    }
} // namespace rsurfaces