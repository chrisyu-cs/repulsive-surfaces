#include "energy/coulomb.h"

namespace rsurfaces
{
    CoulombEnergy::CoulombEnergy(TPEKernel *kernel_, double theta_)
    {
        kernel = kernel_;
        theta = theta_;
        root = 0;
    }

    CoulombEnergy::~CoulombEnergy()
    {
        if (root)
        {
            delete root;
        }
    }

    void CoulombEnergy::Update()
    {
        if (root)
        {
            delete root;
        }
    }

    MeshPtr CoulombEnergy::GetMesh()
    {
        return kernel->mesh;
    }

    GeomPtr CoulombEnergy::GetGeom()
    {
        return kernel->geom;
    }

    Vector2 CoulombEnergy::GetExponents()
    {
        return Vector2{kernel->alpha, kernel->beta};
    }

    BVHNode6D *CoulombEnergy::GetBVH()
    {
        return root;
    }

    double CoulombEnergy::GetTheta()
    {
        return theta;
    }

    double CoulombEnergy::Value()
    {
        double sum = 0;
        #pragma omp parallel for reduction(+ : sum) shared(root)
        for (size_t i = 0; i < kernel->mesh->nVertices(); i++)
        {
            GCVertex v = kernel->mesh->vertex(i);
            double e_v = energyAtVertex(root, v);
            sum += e_v;
        }
        return sum;
    }

    double CoulombEnergy::energyAtVertex(BVHNode6D* node, GCVertex v)
    {
        Vector3 vPos = kernel->geom->inputVertexPositions[v];
        
        double sum = 0;
        for (GCVertex v2 : kernel->mesh->vertices())
        {
            if (v == v2) continue;
            Vector3 vPos2 = kernel->geom->inputVertexPositions[v2];
            double dist = (vPos - vPos2).norm();
            sum += 1 / dist;
        }
        return sum;
    }

    void CoulombEnergy::Differential(Eigen::MatrixXd &output)
    {
        VertexIndices indices = kernel->mesh->getVertexIndices();
        output.setZero();
        Eigen::MatrixXd partialOutput = output;
        #pragma omp parallel firstprivate(partialOutput) shared(root, output)
        {
            #pragma omp for
            for (size_t i = 0; i < kernel->mesh->nVertices(); i++)
            {
                GCVertex v = kernel->mesh->vertex(i);
                Vector3 grad_v = gradientAtVertex(root, v);
                MatrixUtils::addToRow(partialOutput, indices[v], grad_v);
            }

            #pragma omp critical
            {
                output += partialOutput;
            }
        }
    }

    Vector3 CoulombEnergy::gradientAtVertex(BVHNode6D *node, GCVertex v)
    {
        Vector3 vPos = kernel->geom->inputVertexPositions[v];
        Vector3 sum{0, 0, 0};

        for (GCVertex v2 : kernel->mesh->vertices())
        {
            if (v == v2) continue;
            Vector3 vPos2 = kernel->geom->inputVertexPositions[v2];
            Vector3 towardsOther = vPos2 - vPos;
            double dist = towardsOther.norm();
            // Normalize the direction
            towardsOther /= dist;
            // Then divide by r^2 to get the right magnitude
            towardsOther /= dist;
            towardsOther /= dist;

            sum += towardsOther;
        }
        return sum;
    }

} // namespace rsurfaces