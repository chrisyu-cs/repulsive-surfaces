#include "energy/barnes_hut_tpe_6d.h"
#include "helpers.h"
#include "surface_derivatives.h"

namespace rsurfaces
{

BarnesHutTPEnergy6D::BarnesHutTPEnergy6D(TPEKernel *kernel_, BVHNode6D *root_)
{
    kernel = kernel_;
    root = root_;
}

double BarnesHutTPEnergy6D::Value()
{
    double sum = 0;
    for (GCFace f : kernel->mesh->faces())
    {
        sum += computeEnergyOfFace(f, root);
    }
    return sum;
}

double BarnesHutTPEnergy6D::computeEnergyOfFace(GCFace face, BVHNode6D *bvhRoot)
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
        MassPoint mp{mnp.mass, mnp.point, mnp.elementID};
        return kernel->tpe_pair(face, mp);
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


void BarnesHutTPEnergy6D::Differential(Eigen::MatrixXd &output)
{
    std::cout << "TODO: 6D differential" << std::endl;
}

MeshPtr BarnesHutTPEnergy6D::GetMesh()
{
    return kernel->mesh;
}

GeomPtr BarnesHutTPEnergy6D::GetGeom()
{
    return kernel->geom;
}

} // namespace rsurfaces
