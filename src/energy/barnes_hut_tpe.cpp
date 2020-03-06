#include "energy/barnes_hut_tpe.h"
#include "helpers.h"

namespace rsurfaces
{

BarnesHutTPEnergy::BarnesHutTPEnergy(TPEKernel *kernel_, BVHNode3D *root_)
{
    kernel = kernel_;
    root = root_;
}

double BarnesHutTPEnergy::Value()
{
    double sum = 0;
    for (GCFace f : kernel->mesh->faces())
    {
        sum += computeEnergyOfFace(f, root);
    }
    return sum;
}

double BarnesHutTPEnergy::computeEnergyOfFace(GCFace face, BVHNode3D *bvhRoot)
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
    if (bvhRoot->shouldUseCell(bcenter))
    {
        // Use the cluster approximation
        return kernel->tpe_pair(face, bvhRoot->GetMassPoint());
    }
    else
    {
        // Recursively compute it on all children
        double sum = 0;
        for (BVHNode3D *child : bvhRoot->children)
        {
            sum += computeEnergyOfFace(face, child);
        }
        return sum;
    }
}

void BarnesHutTPEnergy::Differential(Eigen::MatrixXd &output)
{
    int rows = 3 * kernel->mesh->nVertices();
    Eigen::VectorXd V, W;
    V.setZero(rows);
    W.setZero(rows);

    // TODO
    return;
}

void BarnesHutTPEnergy::addVOfPair(GCFace face, BVHNode3D *bvhRoot, Eigen::VectorXd &V)
{
    // Each face produces 3 derivatives of the term with that face,
    // corresponding to the three surrounding vertices
    for (GCVertex vert : face.adjacentVertices())
    {
    }
    // TODO: add the term here
}
} // namespace rsurfaces

MeshPtr BarnesHutTPEnergy::GetMesh()
{
    return kernel->mesh;
}

GeomPtr BarnesHutTPEnergy::GetGeom()
{
    return kernel->geom;
}

} // namespace rsurfaces