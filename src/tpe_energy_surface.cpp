#include "tpe_energy_surface.h"
#include "surface_derivatives.h"

namespace rsurfaces
{

SurfaceTPE::SurfaceTPE(MeshPtr m, GeomPtr g, double a, double b)
    : mesh(std::move(m)), geom(std::move(g))
{
    alpha = a;
    beta = b;
}

inline Vector3 faceBarycenter(GeomPtr const &geom, GCFace f)
{
    Vector3 sum{0, 0, 0};
    int count = 0;
    for (GCVertex v : f.adjacentVertices())
    {
        sum += geom->vertexPositions[v];
        count++;
    }
    return sum / count;
}

double SurfaceTPE::tpe_Kf(GCFace f1, GCFace f2) {
    Vector3 n1 = geom->faceNormal(f1);
    Vector3 v1 = faceBarycenter(geom, f1);
    Vector3 v2 = faceBarycenter(geom, f2);

    Vector3 displacement = v1 - v2;
    double numer = pow(dot(n1, displacement), alpha);
    double denom = pow(displacement.norm(), beta);
    return numer / denom;
}

double SurfaceTPE::tpe_pair(GCFace f1, GCFace f2)
{
    double w1 = geom->faceArea(f1);
    double w2 = geom->faceArea(f2);
    return tpe_Kf(f1, f2) * w1 * w2;
}

Vector3 SurfaceTPE::tpe_gradient_Kf(GCFace f1, GCFace f2, GCVertex wrt)
{
    Vector3 n1 = geom->faceNormal(f1);
    Vector3 v1 = faceBarycenter(geom, f1);
    Vector3 v2 = faceBarycenter(geom, f2);
    Vector3 displacement = v1 - v2;

    double A = pow(dot(n1, displacement), alpha);
    double B = pow(displacement.norm(), beta);

    // Derivative of A
    double deriv_A_coeff = alpha * pow(dot(n1, displacement), alpha - 1);

    Jacobian ddx_N = SurfaceDerivs::normalWrtVertex(geom, f1, wrt);
    Jacobian ddx_v1 = SurfaceDerivs::barycenterWrtVertex(f1, wrt);
    Jacobian ddx_v2 = SurfaceDerivs::barycenterWrtVertex(f2, wrt);
    Jacobian ddx_v1_v2 = ddx_v1 - ddx_v2;

    Vector3 deriv_A_prod1 = ddx_N.LeftMultiply(displacement);
    Vector3 deriv_A_prod2 = ddx_v1_v2.LeftMultiply(n1);
    Vector3 deriv_A = deriv_A_coeff * (deriv_A_prod1 + deriv_A_prod2);

    // Derivative of B
    double deriv_B_coeff = beta * pow(displacement.norm(), beta - 1);
    Vector3 disp_normalized = displacement.normalize();
    Vector3 deriv_B = deriv_B_coeff * ddx_v1_v2.LeftMultiply(disp_normalized);

    Vector3 numer = deriv_A * B - A * deriv_B;
    double denom = B * B;
    return numer / denom;
}

Vector3 SurfaceTPE::tpe_gradient_pair(GCFace f1, GCFace f2, GCVertex wrt)
{
    double Kf = tpe_Kf(f1, f2);
    Vector3 grad_Kf = tpe_gradient_Kf(f1, f2, wrt);
    double area1 = geom->faceArea(f1);
    Vector3 grad_area1 = SurfaceDerivs::triangleAreaWrtVertex(geom, f1, wrt);
    double area2 = geom->faceArea(f2);
    Vector3 grad_area2 = SurfaceDerivs::triangleAreaWrtVertex(geom, f2, wrt);

    Vector3 term1 = grad_Kf * area1 * area2;
    Vector3 term2 = Kf * grad_area1 * area2;
    Vector3 term3 = Kf * area1 * grad_area2;

    return term1 + term2 + term3;
}

} // namespace rsurfaces
