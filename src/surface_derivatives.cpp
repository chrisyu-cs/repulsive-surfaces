#include "surface_derivatives.h"

namespace rsurfaces
{
    Jacobian operator+(const Jacobian &a, const Jacobian &b)
    {
        Vector3 col1 = a.col1 + b.col1;
        Vector3 col2 = a.col2 + b.col2;
        Vector3 col3 = a.col3 + b.col3;

        return Jacobian{col1, col2, col3};
    }

    Jacobian operator-(const Jacobian &a, const Jacobian &b)
    {
        return a + (b * -1);
    }

    Jacobian operator-(const Jacobian &a)
    {
        return a * -1;
    }

    Jacobian operator*(const Jacobian &a, double c)
    {
        return Jacobian{a.col1 * c, a.col2 * c, a.col3 * c};
    }

    Jacobian operator*(double c, const Jacobian &a)
    {
        return a * c;
    }

    Jacobian operator/(const Jacobian &a, double c)
    {
        return a * (1.0 / c);
    }

} // namespace rsurfaces