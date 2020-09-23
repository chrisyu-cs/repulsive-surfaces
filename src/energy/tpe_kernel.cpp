#include "energy/tpe_kernel.h"
#include "surface_derivatives.h"
#include "helpers.h"

namespace rsurfaces
{

    TPEKernel::TPEKernel(const MeshPtr &m, const GeomPtr &g, double a, double b)
        : faceBarycenters(*m)
    {
        mesh = m;
        geom = g;
        alpha = a;
        beta = b;
    }

    void TPEKernel::recomputeBarycenters()
    {
        for (GCFace f : mesh->faces())
        {
            faceBarycenters[f] = faceBarycenter(geom, f);
        }
    }

} // namespace rsurfaces
