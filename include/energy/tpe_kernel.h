#pragma once

#include "spatial/bvh_types.h"
#include "surface_derivatives.h"
#include "helpers.h"

namespace rsurfaces
{
    using namespace geometrycentral;

    class TPEKernel
    {
    public:
        TPEKernel(const MeshPtr &m, const GeomPtr &g, double alpha, double beta);

        double tpe_Kf(Vector3 v1, Vector3 v2, Vector3 n1);

        template <typename F1, typename F2>
        double tpe_Kf(F1 f1, F2 f2);

        template <typename F1, typename F2>
        Vector3 tpe_gradient_pair(F1 f1, F2 f2, GCVertex wrt);

        template <typename F1, typename F2>
        double tpe_pair(F1 f1, F2 f2);

        void numericalTest();

        MeshPtr mesh;
        GeomPtr geom;
        double alpha, beta;

    private:
        template <typename F1, typename F2>
        Vector3 tpe_gradient_Kf(F1 f1, F2 f2, GCVertex wrt);

        Vector3 tpe_gradient_pair_num(GCFace f1, GCFace f2, GCVertex wrt, double eps);
    };

    template <typename F1, typename F2>
    double TPEKernel::tpe_Kf(F1 f1, F2 f2)
    {
        Vector3 n1 = faceNormal(geom, f1);
        Vector3 v1 = faceBarycenter(geom, f1);
        Vector3 v2 = faceBarycenter(geom, f2);
        return tpe_Kf(v1, v2, n1);
    }

    template <>
    inline double TPEKernel::tpe_Kf(GCFace f1, GCFace f2)
    {
        if (f1 == f2)
        {
            return 0;
        }
        Vector3 n1 = faceNormal(geom, f1);
        Vector3 v1 = faceBarycenter(geom, f1);
        Vector3 v2 = faceBarycenter(geom, f2);
        return tpe_Kf(v1, v2, n1);
    }

    template <typename F1, typename F2>
    double TPEKernel::tpe_pair(F1 f1, F2 f2)
    {
        double w1 = faceArea(geom, f1);
        double w2 = faceArea(geom, f2);
        return tpe_Kf(f1, f2) * w1 * w2;
    }

    template <typename F1, typename F2>
    Vector3 TPEKernel::tpe_gradient_pair(F1 f1, F2 f2, GCVertex wrt)
    {
        Vector3 p1 = faceBarycenter(geom, f1);
        Vector3 p2 = faceBarycenter(geom, f2);
        Vector3 n1 = faceNormal(geom, f1);

        double Kf = tpe_Kf(p1, p2, n1);
        Vector3 grad_Kf = tpe_gradient_Kf(f1, f2, wrt);
        double area1 = faceArea(geom, f1);
        Vector3 grad_area1 = SurfaceDerivs::triangleAreaWrtVertex(geom, f1, wrt);
        double area2 = faceArea(geom, f2);
        Vector3 grad_area2 = SurfaceDerivs::triangleAreaWrtVertex(geom, f2, wrt);

        Vector3 term1 = grad_Kf * area1 * area2;
        Vector3 term2 = Kf * grad_area1 * area2;
        Vector3 term3 = Kf * area1 * grad_area2;

        return term1 + term2 + term3;
    }

    template <typename F1, typename F2>
    Vector3 TPEKernel::tpe_gradient_Kf(F1 f1, F2 f2, GCVertex wrt)
    {
        Vector3 n1 = faceNormal(geom, f1);
        Vector3 v1 = faceBarycenter(geom, f1);
        Vector3 v2 = faceBarycenter(geom, f2);
        Vector3 displacement = v1 - v2;

        double dot_nD = dot(n1, displacement);
        double A = pow(fabs(dot_nD), alpha);
        double B = pow(displacement.norm(), beta);

        // Derivative of A
        double deriv_A_coeff = alpha * pow(fabs(dot_nD), alpha - 1);
        double sgn_dot = sgn_fn(dot_nD);

        Jacobian ddx_N = SurfaceDerivs::normalWrtVertex(geom, f1, wrt);
        Jacobian ddx_v1 = SurfaceDerivs::barycenterWrtVertex(f1, wrt);
        Jacobian ddx_v2 = SurfaceDerivs::barycenterWrtVertex(f2, wrt);
        Jacobian ddx_v1_v2 = ddx_v1 - ddx_v2;

        Vector3 deriv_A_prod1 = ddx_N.LeftMultiply(displacement);
        Vector3 deriv_A_prod2 = ddx_v1_v2.LeftMultiply(n1);
        Vector3 deriv_A = deriv_A_coeff * sgn_dot * (deriv_A_prod1 + deriv_A_prod2);

        // Derivative of B
        double deriv_B_coeff = beta * pow(displacement.norm(), beta - 1);
        Vector3 disp_normalized = displacement.normalize();
        Vector3 deriv_B = deriv_B_coeff * ddx_v1_v2.LeftMultiply(disp_normalized);

        Vector3 numer = deriv_A * B - A * deriv_B;
        double denom = B * B;
        return numer / denom;
    }

} // namespace rsurfaces
