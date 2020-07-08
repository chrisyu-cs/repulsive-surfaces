#include "spatial/convolution_kernel.h"

namespace rsurfaces
{
    RieszKernel::RieszKernel(double s_) {
        s = s_;
    }

    double RieszKernel::Coefficient(Vector3 x, Vector3 y)
    {
        if (x == y) return 0;
        std::cout << "Distance = " << norm(x - y) << ", value = " << 1.0 / pow(norm(x - y), s) << std::endl;
        return 1.0 / pow(norm(x - y), s);
    }
} // namespace rsurfaces