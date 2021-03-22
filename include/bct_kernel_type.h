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
            b.segment(nV3, nConstraints) += C * v.segment(0, nV3);
            // C^T * lambda goes into the top block of b
            b.segment(0, nV3) += C.transpose() * v.segment(nV3, nConstraints);
        }
        else
        {
            // C * v goes into the bottom block of b
            b.segment(nV3, nConstraints) = C * v.segment(0, nV3);
            // C^T * lambda goes into the top block of b
            b.segment(0, nV3) = C.transpose() * v.segment(nV3, nConstraints);
        }
    }
} // namespace rsurfaces