#include "energy/tpe_kernel.h"
#include "surface_derivatives.h"
#include "helpers.h"

namespace rsurfaces
{

TPEKernel::TPEKernel(const MeshPtr &m, const GeomPtr &g, double a, double b)
{
    mesh = m;
    geom = g;
    alpha = a;
    beta = b;
}

double TPEKernel::tpe_Kf(Vector3 v1, Vector3 v2, Vector3 n1)
{
    Vector3 displacement = v1 - v2;
    double numer = pow(fabs(dot(n1, displacement)), alpha);
    double denom = pow(displacement.norm(), beta);
    return numer / denom;
}

double TPEKernel::tpe_Kf(GCFace f1, GCFace f2)
{
    if (f1 == f2)
    {
        return 0;
    }
    Vector3 n1 = geom->faceNormal(f1);
    Vector3 v1 = faceBarycenter(geom, f1);
    Vector3 v2 = faceBarycenter(geom, f2);
    return tpe_Kf(v1, v2, n1);
}

double TPEKernel::tpe_pair(GCFace f1, GCFace f2)
{
    double w1 = geom->faceArea(f1);
    double w2 = geom->faceArea(f2);
    return tpe_Kf(f1, f2) * w1 * w2;
}

double TPEKernel::tpe_pair(GCFace f1, MassPoint p2)
{
    double w1 = geom->faceArea(f1);
    double w2 = p2.mass;
    Vector3 v1 = faceBarycenter(geom, f1);
    Vector3 n1 = geom->faceNormal(f1);
    return tpe_Kf(v1, p2.point, n1) * w1 * w2;
}

inline int sgn_fn(double x)
{
    if (x > 0)
        return 1;
    else if (x < 0)
        return -1;
    else
        return 0;
}

Vector3 TPEKernel::tpe_gradient_Kf(GCFace f1, GCFace f2, GCVertex wrt)
{
    Vector3 n1 = geom->faceNormal(f1);
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

Vector3 TPEKernel::tpe_gradient_Kf(GCFace f1, MassNormalPoint f2, GCVertex wrt)
{
    // Same as normal case, but derivatives of f2 wrt the vertex are all 0.
    Vector3 n1 = geom->faceNormal(f1);
    Vector3 v1 = faceBarycenter(geom, f1);
    Vector3 v2 = f2.point;
    Vector3 displacement = v1 - v2;

    double dot_nD = dot(n1, displacement);
    double A = pow(fabs(dot_nD), alpha);
    double B = pow(displacement.norm(), beta);

    // Derivative of A
    double deriv_A_coeff = alpha * pow(fabs(dot_nD), alpha - 1);
    double sgn_dot = sgn_fn(dot_nD);

    Jacobian ddx_N = SurfaceDerivs::normalWrtVertex(geom, f1, wrt);
    Jacobian ddx_v1 = SurfaceDerivs::barycenterWrtVertex(f1, wrt);

    Vector3 deriv_A_prod1 = ddx_N.LeftMultiply(displacement);
    Vector3 deriv_A_prod2 = ddx_v1.LeftMultiply(n1);
    Vector3 deriv_A = deriv_A_coeff * sgn_dot * (deriv_A_prod1 + deriv_A_prod2);

    // Derivative of B
    double deriv_B_coeff = beta * pow(displacement.norm(), beta - 1);
    Vector3 disp_normalized = displacement.normalize();
    Vector3 deriv_B = deriv_B_coeff * ddx_v1.LeftMultiply(disp_normalized);

    Vector3 numer = deriv_A * B - A * deriv_B;
    double denom = B * B;
    return numer / denom;
}

Vector3 TPEKernel::tpe_gradient_Kf(MassNormalPoint f1, GCFace f2, GCVertex wrt)
{
    // Same as normal case, but derivatives of f1 wrt the vertex are all 0.
    Vector3 n1 = f1.normal;
    Vector3 v1 = f1.point;
    Vector3 v2 = faceBarycenter(geom, f2);
    Vector3 displacement = v1 - v2;

    double dot_nD = dot(n1, displacement);
    double A = pow(fabs(dot_nD), alpha);
    double B = pow(displacement.norm(), beta);

    // Derivative of A
    double deriv_A_coeff = alpha * pow(fabs(dot_nD), alpha - 1);
    double sgn_dot = sgn_fn(dot_nD);

    Jacobian ddx_v2 = SurfaceDerivs::barycenterWrtVertex(f2, wrt);
    Jacobian ddx_neg_v2 = -1 * ddx_v2;

    Vector3 deriv_A_prod2 = ddx_neg_v2.LeftMultiply(n1);
    Vector3 deriv_A = deriv_A_coeff * sgn_dot * deriv_A_prod2;

    // Derivative of B
    double deriv_B_coeff = beta * pow(displacement.norm(), beta - 1);
    Vector3 disp_normalized = displacement.normalize();
    Vector3 deriv_B = deriv_B_coeff * ddx_neg_v2.LeftMultiply(disp_normalized);

    Vector3 numer = deriv_A * B - A * deriv_B;
    double denom = B * B;
    return numer / denom;
}

void TPEKernel::numericalTest()
{
    double avg = 0;
    int count = 0;
    double max_err = 0;

    double max_norm_a = 0;
    double max_norm_n = 0;

    for (GCFace f1 : mesh->faces())
    {
        for (GCFace f2 : mesh->faces())
        {
            if (f1 == f2)
                continue;
            GCVertex vert = f2.halfedge().vertex();

            Vector3 grad_num = tpe_gradient_pair_num(f1, f2, vert, 0.001);
            Vector3 grad_a = tpe_gradient_pair(f1, f2, vert);

            double pct_diff = 100 * norm(grad_num - grad_a) / norm(grad_num);
            avg += pct_diff;

            if (pct_diff > max_err)
            {
                std::cout << "Analytic =  " << grad_a << std::endl;
                std::cout << "Numerical = " << grad_num << std::endl;
            }

            max_err = fmax(max_err, pct_diff);
            max_norm_a = fmax(max_norm_a, norm(grad_a));
            max_norm_n = fmax(max_norm_n, norm(grad_num));

            count++;
        }
    }

    avg /= count;
    std::cout << "Max analytic norm = " << max_norm_a << ", numerical = " << max_norm_n << std::endl;
    std::cout << "max diff = " << max_err << " percent" << std::endl;
    std::cout << "average relative diff = " << avg << " percent" << std::endl;
}

Vector3 TPEKernel::tpe_Kf_partial_wrt_v1(Vector3 v1, Vector3 v2, Vector3 n1)
{
    double n_dot = dot(n1, v1 - v2);
    double A = pow(fabs(n_dot), alpha);
    double front_dA = alpha * pow(fabs(n_dot), alpha - 1) * sgn_fn(n_dot);
    Vector3 deriv_A = front_dA * n1;

    double norm_dist = norm(v1 - v2);
    double B = pow(norm_dist, beta);
    Vector3 deriv_B = beta * pow(norm_dist, beta - 1) * ((v1 - v2) / norm_dist);

    return (deriv_A * B - deriv_B * A) / (B * B);
}

Vector3 TPEKernel::tpe_Kf_partial_wrt_v2(Vector3 v1, Vector3 v2, Vector3 n1)
{
    return -tpe_Kf_partial_wrt_v1(v1, v2, n1);
}

Vector3 TPEKernel::tpe_Kf_partial_wrt_n1(Vector3 v1, Vector3 v2, Vector3 n1)
{
    double n_dot = dot(n1, v1 - v2);
    double A = pow(fabs(n_dot), alpha);
    double front_dA = alpha * pow(fabs(n_dot), alpha - 1) * sgn_fn(n_dot);
    Vector3 deriv_A = front_dA * (v1 - v2);
    double B = pow(norm(v1 - v2), beta);
    return deriv_A / B;
}

Vector3 TPEKernel::tpe_gradient_pair(GCFace f1, GCFace f2, GCVertex wrt)
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

Vector3 TPEKernel::tpe_gradient_pair(GCFace f1, MassNormalPoint f2, GCVertex wrt)
{
    // In this case, we can assume f2 is not near the vertex,
    // so derivatives of f2 are 0.
    double Kf = tpe_Kf(faceBarycenter(geom, f1), f2.point, geom->faceNormal(f1));
    Vector3 grad_Kf = tpe_gradient_Kf(f1, f2, wrt);
    double area1 = geom->faceArea(f1);
    Vector3 grad_area1 = SurfaceDerivs::triangleAreaWrtVertex(geom, f1, wrt);
    double area2 = f2.mass;

    Vector3 term1 = grad_Kf * area1 * area2;
    Vector3 term2 = Kf * grad_area1 * area2;

    return term1 + term2;
}

Vector3 TPEKernel::tpe_gradient_pair(MassNormalPoint f1, GCFace f2, GCVertex wrt)
{
    double Kf = tpe_Kf(f1.point, faceBarycenter(geom, f2), f1.normal);
    Vector3 grad_Kf = tpe_gradient_Kf(f1, f2, wrt);
    double area1 = f1.mass;
    double area2 = geom->faceArea(f2);
    Vector3 grad_area2 = SurfaceDerivs::triangleAreaWrtVertex(geom, f2, wrt);

    Vector3 term1 = grad_Kf * area1 * area2;
    Vector3 term3 = Kf * area1 * grad_area2;

    return term1 + term3;
}

Vector3 TPEKernel::tpe_gradient_pair_num(GCFace f1, GCFace f2, GCVertex wrt, double eps)
{
    double origEnergy = tpe_pair(f1, f2);
    Vector3 origPos = geom->inputVertexPositions[wrt];

    geom->inputVertexPositions[wrt] = origPos + Vector3{eps, 0, 0};
    double energy_x = tpe_pair(f1, f2);
    geom->inputVertexPositions[wrt] = origPos + Vector3{0, eps, 0};
    double energy_y = tpe_pair(f1, f2);
    geom->inputVertexPositions[wrt] = origPos + Vector3{0, 0, eps};
    double energy_z = tpe_pair(f1, f2);
    geom->inputVertexPositions[wrt] = origPos;

    double dx = (energy_x - origEnergy) / eps;
    double dy = (energy_y - origEnergy) / eps;
    double dz = (energy_z - origEnergy) / eps;

    return Vector3{dx, dy, dz};
}

} // namespace rsurfaces
