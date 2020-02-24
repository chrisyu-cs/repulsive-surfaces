#include "tpe_energy_surface.h"
#include "surface_derivatives.h"

namespace rsurfaces
{

TPEKernel::TPEKernel(MeshPtr m, GeomPtr g, double a, double b)
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
        sum += geom->inputVertexPositions[v];
        count++;
    }
    return sum / count;
}

double TPEKernel::tpe_Kf(GCFace f1, GCFace f2) {
    Vector3 n1 = geom->faceNormal(f1);
    Vector3 v1 = faceBarycenter(geom, f1);
    Vector3 v2 = faceBarycenter(geom, f2);

    Vector3 displacement = v1 - v2;
    double numer = pow(dot(n1, displacement), alpha);
    double denom = pow(displacement.norm(), beta);
    return numer / denom;
}

double TPEKernel::tpe_pair(GCFace f1, GCFace f2)
{
    double w1 = geom->faceArea(f1);
    double w2 = geom->faceArea(f2);
    return tpe_Kf(f1, f2) * w1 * w2;
}

Vector3 TPEKernel::tpe_gradient_Kf(GCFace f1, GCFace f2, GCVertex wrt)
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

Vector3 TPEKernel::tpe_gradient_Kf_num(GCFace f1, GCFace f2, GCVertex wrt, double eps) {
    double origEnergy = tpe_Kf(f1, f2);
    Vector3 origPos = geom->inputVertexPositions[wrt];

    geom->inputVertexPositions[wrt] = origPos + Vector3{eps, 0, 0};
    double energy_x = tpe_Kf(f1, f2);
    geom->inputVertexPositions[wrt] = origPos + Vector3{0, eps, 0};
    double energy_y = tpe_Kf(f1, f2);
    geom->inputVertexPositions[wrt] = origPos + Vector3{0, 0, eps};
    double energy_z = tpe_Kf(f1, f2);
    geom->inputVertexPositions[wrt] = origPos;

    double dx = (energy_x - origEnergy) / eps;
    double dy = (energy_y - origEnergy) / eps;
    double dz = (energy_z - origEnergy) / eps;

    return Vector3{dx, dy, dz};
}

void TPEKernel::numericalTest() {
    double avg = 0;
    int count = 0;
    double max_err = 0;

    for (GCFace f1 : mesh->faces()) {
        for (GCFace f2 : mesh->faces()) {
            if (f1 == f2) continue;
            GCVertex vert = f2.halfedge().vertex();

            Vector3 grad_num = tpe_gradient_Kf_num(f1, f2, vert, 0.001);
            Vector3 grad_a = tpe_gradient_Kf(f1, f2, vert);

            double pct_diff = 100 * norm(grad_num - grad_a) / norm(grad_num);
            avg += pct_diff;

            if (pct_diff > max_err) {
                std::cout << "Analytic =  " << grad_a << std::endl;
                std::cout << "Numerical = " << grad_num << std::endl;
            }

            max_err = fmax(max_err, pct_diff);

            count++;
        }
    }

    avg /= count;
    std::cout << "max diff = " << max_err << " percent" << std::endl;
    std::cout << "average relative diff = " << avg << " percent" << std::endl;
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

} // namespace rsurfaces
