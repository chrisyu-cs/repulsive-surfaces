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
        double denom = pow(displacement.norm2(), beta / 2);
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

    Vector3 TPEKernel::tpe_gradient_cluster_center(GCFace f1, MassNormalPoint f2) {
        return Vector3{0, 0, 0};
    }

    double TPEKernel::tpe_gradient_cluster_mass(GCFace f1, MassNormalPoint f2) {
        return 0;
    }

} // namespace rsurfaces
