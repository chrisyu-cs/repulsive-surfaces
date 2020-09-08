#include "energy/barnes_hut_tpe_xdiff.h"
#include "helpers.h"
#include "surface_derivatives.h"

namespace rsurfaces
{

    BarnesHutTPEnergyXDiff::BarnesHutTPEnergyXDiff(TPEKernel *kernel_, double theta_)
    {
        kernel = kernel_;
        theta = theta_;
        root = 0;
    }

    double BarnesHutTPEnergyXDiff::Value()
    {
        if (!root)
        {
            std::cerr << "BVH for BarnesHutTPEnergyXDiff was not initialized. Call Update() first." << std::endl;
            std::exit(1);
        }

        double sum = 0;
        for (GCFace f : kernel->mesh->faces())
        {
            sum += computeEnergyOfFace(f, root);
        }
        return sum;
    }

    double BarnesHutTPEnergyXDiff::computeEnergyOfFace(GCFace face, BVHNode6D *bvhRoot)
    {
        Vector3 bcenter = faceBarycenter(kernel->geom, face);

        if (bvhRoot->nodeType == BVHNodeType::Empty)
        {
            // Empty cluster = no value
            return 0;
        }
        else if (bvhRoot->nodeType == BVHNodeType::Leaf)
        {
            // Compute the energy exactly for the one face in the cluster
            GCFace f2 = bvhRoot->getSingleFace(kernel->mesh);
            return kernel->tpe_pair(face, f2);
        }
        if (bvhRoot->isAdmissibleFrom(bcenter))
        {
            // Use the cluster approximation
            MassNormalPoint mnp = bvhRoot->GetMassNormalPoint();
            return kernel->tpe_pair(face, mnp);
        }
        else
        {
            // Recursively compute it on all children
            double sum = 0;
            for (BVHNode6D *child : bvhRoot->children)
            {
                sum += computeEnergyOfFace(face, child);
            }
            return sum;
        }
    }

    void BarnesHutTPEnergyXDiff::Differential(Eigen::MatrixXd &output)
    {
        if (!root)
        {
            std::cerr << "BVH for BarnesHutTPEnergyXDiff was not initialized. Call Update() first." << std::endl;
            std::exit(1);
        }

        // TODO: finish this (percolate diffs downward)
    }

    void BarnesHutTPEnergyXDiff::accClusterGradients(BVHNode6D *bvhRoot, GCFace face, DataTreeContainer<BHDiffData> *data)
    {
        Vector3 bcenter = faceBarycenter(kernel->geom, face);

        if (bvhRoot->nodeType == BVHNodeType::Empty)
        {
            // Empty cluster = no value
        }
        else
        {
            if (bvhRoot->isAdmissibleFrom(bcenter))
            {
                DataTree<BHDiffData> *dataNode = data->GetDataNode(bvhRoot);
                Vector3 Z_V = bvhRoot->centerOfMass * bvhRoot->totalMass;
                double A_V = bvhRoot->totalMass;
                // Fill in the data for this (face, cluster) paid
                dataNode->data.dCenter = diffEnergyWrtSumCenters(face, Z_V, A_V);
                dataNode->data.dArea = diffEnergyWrtSumAreas(face, Z_V, A_V);
            }
            else
            {
                // Recursively compute it on all children
                for (BVHNode6D *child : bvhRoot->children)
                {
                    accClusterGradients(child, face, data);
                }
            }
        }
    }

    // Differentiating the TPE kernel by Z_V (weighted sum of triangle centers)
    Vector3 BarnesHutTPEnergyXDiff::diffKernelWrtSumCenters(Vector3 x_T, Vector3 n_T, Vector3 Z_V, double A_V)
    {
        // Displacement vector between vertex and cluster
        Vector3 disp = x_T - Z_V / A_V;
        double normDisp = norm(disp);
        // Value of the dot product w/ normal
        double p = dot(n_T, disp);
        // Numerator and denominator of TPE kernel
        double A = pow(fabs(p), kernel->alpha);
        double B = pow(normDisp, kernel->beta);
        // Weight on diagonal of Jacobian of -Z_V / A_V wrt Z_V
        double jWeight = -1.0 / A_V;

        Vector3 diff_A = (kernel->alpha * pow(fabs(p), kernel->alpha - 1)) * sgn_fn(p) * jWeight * n_T;
        Vector3 diff_B = (kernel->beta * pow(normDisp, kernel->beta - 1)) * (disp / normDisp) * jWeight;

        // Quotient rule for A / B
        return (diff_A * B - diff_B * A) / (B * B);
    }

    Vector3 BarnesHutTPEnergyXDiff::diffEnergyWrtSumCenters(GCFace face, Vector3 Z_V, double A_V)
    {
        // Neither A_T or A_V directly produce a derivative with Z_V
        Vector3 x_T = faceBarycenter(kernel->geom, face);
        Vector3 n_T = faceNormal(kernel->geom, face);
        double a_T = faceArea(kernel->geom, face);
        Vector3 diff_k = diffKernelWrtSumCenters(x_T, n_T, Z_V, A_V);
        return diff_k * a_T * A_V;
    }

    // Differentiating the TPE kernel by A_V (sum of triangle areas)
    double BarnesHutTPEnergyXDiff::diffKernelWrtSumAreas(Vector3 x_T, Vector3 n_T, Vector3 Z_V, double A_V)
    {
        // Displacement vector between vertex and cluster
        Vector3 disp = x_T - Z_V / A_V;
        double normDisp = norm(disp);
        // Value of the dot product w/ normal
        double p = dot(n_T, disp);
        // Numerator and denominator of TPE kernel
        double A = pow(fabs(p), kernel->alpha);
        double B = pow(normDisp, kernel->beta);

        Vector3 diffCenter = (Z_V / (A_V * A_V));

        double diff_A = (kernel->alpha * pow(fabs(p), kernel->alpha - 1)) * sgn_fn(p) * dot(diffCenter, n_T);
        double diff_B = (kernel->beta * pow(normDisp, kernel->beta - 1)) * dot(disp / normDisp, diffCenter);

        // Quotient rule for A / B
        return (diff_A * B - diff_B * A) / (B * B);
    }

    double BarnesHutTPEnergyXDiff::diffEnergyWrtSumAreas(GCFace face, Vector3 Z_V, double A_V)
    {
        Vector3 x_T = faceBarycenter(kernel->geom, face);
        Vector3 n_T = faceNormal(kernel->geom, face);
        double a_T = faceArea(kernel->geom, face);
        Vector3 bcenter = Z_V / A_V;
        // Evaluate K with face center, cluster barycenter, and face normal
        double K = kernel->tpe_Kf(x_T, bcenter, n_T);

        // Apply product rule to K * A_T * A_V
        // = A_T * (dK * A_V + K * 1) = A_T * (dK * A_V + K)
        double diff_K = diffKernelWrtSumAreas(x_T, n_T, Z_V, A_V);
        return a_T * (diff_K * A_V + K);
    }

    void BarnesHutTPEnergyXDiff::Update()
    {
        if (root)
        {
            delete root;
        }
        root = Create6DBVHFromMeshFaces(kernel->mesh, kernel->geom, theta);
    }

    MeshPtr BarnesHutTPEnergyXDiff::GetMesh()
    {
        return kernel->mesh;
    }

    GeomPtr BarnesHutTPEnergyXDiff::GetGeom()
    {
        return kernel->geom;
    }

    Vector2 BarnesHutTPEnergyXDiff::GetExponents()
    {
        return Vector2{kernel->alpha, kernel->beta};
    }

    BVHNode6D *BarnesHutTPEnergyXDiff::GetBVH()
    {
        return root;
    }
} // namespace rsurfaces