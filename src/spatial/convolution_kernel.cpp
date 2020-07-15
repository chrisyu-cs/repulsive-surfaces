#include "spatial/convolution_kernel.h"

namespace rsurfaces
{
    RieszKernel::RieszKernel(double s_)
    {
        s = s_;
    }

    double RieszKernel::Coefficient(Vector3 x, Vector3 y)
    {
        if (norm2(x - y) < 1e-3)
        {
            return 0;
        }
        return 1.0 / pow(norm(x - y), s);
    }
} // namespace rsurfaces