#include "energy/barnes_hut_tpe.h"
#include "helpers.h"
#include "surface_derivatives.h"

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
    if (bvhRoot->isAdmissibleFrom(bcenter))
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
    Eigen::MatrixXd V, W;
    V.setZero(kernel->mesh->nVertices(), 3);
    W.setZero(kernel->mesh->nVertices(), 3);

    VertexIndices indices = kernel->mesh->getVertexIndices();

    // Add the V and W values
    addV(root, V, indices);
    addW(root, W, indices);

    std::cout << "Norm V = " << V.norm() << std::endl;
    std::cout << "Norm W = " << W.norm() << std::endl;

    output = V + W;
}

void BarnesHutTPEnergy::addV(BVHNode3D *bvhRoot, Eigen::MatrixXd &V, VertexIndices &indices)
{
    for (GCFace face : kernel->mesh->faces())
    {
        addVOfFace(face, bvhRoot, V, indices);
    }
}

void BarnesHutTPEnergy::addVOfFace(GCFace face, BVHNode3D *node, Eigen::MatrixXd &V, VertexIndices &indices)
{
    if (node->nodeType == BVHNodeType::Empty)
    {
        return;
    }
    else
    {
        Vector3 pos_I = faceBarycenter(kernel->geom, face);
        // If the cluster is admissible, accumulate the values
        if (node->isAdmissibleFrom(pos_I))
        {
            // This works for both leaf and cluster -- in the leaf case, all
            // the values just come from the one triangle
            Vector3 normal_I = kernel->geom->faceNormal(face);
            double l_I = kernel->geom->faceArea(face);
            // Both mass and center of mass are defined for both leaf and cluster
            double K = kernel->tpe_Kf(pos_I, node->centerOfMass, normal_I);
            double l_N = node->totalMass;

            Vector3 dK_dxI = kernel->tpe_Kf_partial_wrt_v1(pos_I, node->centerOfMass, normal_I);
            Vector3 dK_dNI = kernel->tpe_Kf_partial_wrt_n1(pos_I, node->centerOfMass, normal_I);

            for (GCVertex vert : face.adjacentVertices())
            {
                Jacobian dNI_dGammaI = SurfaceDerivs::normalWrtVertex(kernel->geom, face, vert);
                Vector3 dlI_dGammaI = SurfaceDerivs::triangleAreaWrtVertex(kernel->geom, face, vert);

                Vector3 v_vert = l_N * (K * dlI_dGammaI + l_I * dK_dxI + l_I * dNI_dGammaI.LeftMultiply(dK_dNI));
                size_t base_index = indices[vert];

                V(base_index, 0) += v_vert.x;
                V(base_index, 1) += v_vert.y;
                V(base_index, 2) += v_vert.z;
            }
        }
        // Otherwise, continue traversing to admissible nodes
        else
        {
            for (BVHNode3D *child : node->children)
            {
                addVOfFace(face, child, V, indices);
            }
        }
    }
}

void BarnesHutTPEnergy::addW(BVHNode3D *bvhRoot, Eigen::MatrixXd &W, VertexIndices &indices)
{
    size_t nNodes = bvhRoot->numNodesInBranch;
    std::vector<double> xi(nNodes);
    std::vector<Vector3> eta(nNodes);

    for (size_t i = 0; i < nNodes; i++)
    {
        xi[i] = 0;
        eta[i] = Vector3{0, 0, 0};
    }

    // Add all the values for all faces
    for (GCFace face : kernel->mesh->faces())
    {
        accumulateWValues(face, bvhRoot, xi, eta);
    }
    // Loop over all clusters and accumulate into the global vector
    addWForAllClusters(bvhRoot, W, xi, eta, indices);
}

void BarnesHutTPEnergy::addWForAllClusters(BVHNode3D *node, Eigen::MatrixXd &W, std::vector<double> &xi,
                                           std::vector<Vector3> &eta, VertexIndices &indices)
{
    size_t i = node->nodeID;
    if (xi[i] != 0 || eta[i] != Vector3{0, 0, 0})
    {
        // Multiply in the values
        std::vector<GCFace> clusterFaces;
        node->addAllFaces(kernel->mesh, clusterFaces);
        for (GCFace face : clusterFaces)
        {
            for (GCVertex vert : face.adjacentVertices())
            {
                Vector3 dlN_dGammaN = SurfaceDerivs::triangleAreaWrtVertex(kernel->geom, face, vert);
                // Derivative of single face barycenter wrt single vertex position
                Jacobian dBN_dGammaN = SurfaceDerivs::barycenterWrtVertex(face, vert);
                // Derivative of cluster barycenter wrt single face barycenter
                double dxN_dBN = kernel->geom->faceArea(face) / node->totalMass;
                Jacobian dxN_dGammaN = (node->totalMass * dxN_dBN) * dBN_dGammaN;

                size_t base_i = indices[vert];
                // Multiply and add to each vertex index
                Vector3 increment = xi[i] * dlN_dGammaN + dxN_dGammaN.LeftMultiply(eta[i]);
                W(base_i, 0) += increment.x;
                W(base_i, 1) += increment.y;
                W(base_i, 2) += increment.z;
            }
        }
    }

    // If this node has children, process those too
    if (node->nodeType == BVHNodeType::Interior)
    {
        for (BVHNode3D *child : node->children)
        {
            addWForAllClusters(child, W, xi, eta, indices);
        }
    }
}

void BarnesHutTPEnergy::accumulateWValues(GCFace face, BVHNode3D *node, std::vector<double> &xi,
                                          std::vector<Vector3> &eta)
{
    if (node->nodeType == BVHNodeType::Empty)
    {
        return;
    }
    else
    {
        Vector3 pos_I = faceBarycenter(kernel->geom, face);
        // If the current node is admissible, add the values
        if (node->isAdmissibleFrom(pos_I))
        {
            // This works for both leaves and clusters
            size_t node_id = node->nodeID;
            double l_I = kernel->geom->faceArea(face);
            Vector3 normal_I = kernel->geom->faceNormal(face);
            double K = kernel->tpe_Kf(pos_I, node->centerOfMass, normal_I);

            xi[node_id] += l_I * K;
            eta[node_id] += l_I * kernel->tpe_Kf_partial_wrt_v2(pos_I, node->centerOfMass, normal_I);
        }
        // Otherwise, keep traversing until we find admissible nodes
        else
        {
            for (BVHNode3D *child : node->children)
            {
                accumulateWValues(face, child, xi, eta);
            }
        }
    }
}

MeshPtr BarnesHutTPEnergy::GetMesh()
{
    return kernel->mesh;
}

GeomPtr BarnesHutTPEnergy::GetGeom()
{
    return kernel->geom;
}

} // namespace rsurfaces