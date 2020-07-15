#include "energy/barnes_hut_tpe_6d.h"
#include "helpers.h"
#include "surface_derivatives.h"

namespace rsurfaces
{

    BarnesHutTPEnergy6D::BarnesHutTPEnergy6D(TPEKernel *kernel_, double theta_)
    {
        kernel = kernel_;
        theta = theta_;
        root = 0;
    }

    double BarnesHutTPEnergy6D::Value()
    {
        if (!root) {
            std::cerr << "BVH for BarnesHutTPEnergy6D was not initialized. Call Update() first." << std::endl;
            std::exit(1);
        }
        
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
        if (!root) {
            std::cerr << "BVH for BarnesHutTPEnergy6D was not initialized. Call Update() first." << std::endl;
            std::exit(1);
        }
        
        VertexIndices indices = kernel->mesh->getVertexIndices();

        for (GCFace f : kernel->mesh->faces())
        {
            accumulateTPEGradient(output, root, f, indices);
        }
    }

    inline void addToRow(Eigen::MatrixXd &M, size_t row, Vector3 v)
    {
        M(row, 0) += v.x;
        M(row, 1) += v.y;
        M(row, 2) += v.z;
    }

    // Add derivatives of all energy terms of the form (f1, _) or (_, f1)
    // with respect to the neighbor vertices of f1.
    void BarnesHutTPEnergy6D::accumulateTPEGradient(Eigen::MatrixXd &gradients, BVHNode6D *node, GCFace face1,
                                                    surface::VertexData<size_t> indices)
    {
        if (node->nodeType == BVHNodeType::Empty)
        {
            return;
        }
        else if (node->nodeType == BVHNodeType::Leaf)
        {
            // If this is a leaf, then it only has one face in it, so just use it
            GCFace face2 = node->getSingleFace(kernel->mesh);
            // Skip if the faces are the same
            if (face1 == face2)
                return;

            // Differentiate by adjacent vertices to face1
            std::vector<GCVertex> neighborVerts;

            for (GCVertex v : face1.adjacentVertices())
            {
                // Add the forward term (f1, f2)
                Vector3 deriv1 = kernel->tpe_gradient_pair(face1, face2, v);
                size_t r = indices[v];
                addToRow(gradients, r, deriv1);

                // Determine if the reverse term (f2, f1) should be added.
                // If v is also adjacent to f2, then it shouldn't be, because
                // (f2, f1) wrt v will be accumulated later.
                bool noOverlap = true;
                for (GCVertex v2 : face2.adjacentVertices())
                {
                    if (v == v2)
                        noOverlap = false;
                }
                // If v is not adjacent to v2, then we need to add (f2, f1) wrt v now.
                if (noOverlap)
                {
                    Vector3 deriv2 = kernel->tpe_gradient_pair(face2, face1, v);
                    addToRow(gradients, r, deriv2);
                }
            }
        }
        else
        {
            Vector3 f1_center = faceBarycenter(kernel->geom, face1);
            if (node->isAdmissibleFrom(f1_center))
            {
                Vector3 normal = node->averageNormal;
                // This cell is far enough away that we can treat it as a single body
                MassNormalPoint mnp2 = node->GetMassNormalPoint();

                // Differentiate both terms for all neighbors
                for (GCVertex v : face1.adjacentVertices())
                {
                    // Derivatives of both foward and reverse terms
                    addToRow(gradients, indices[v], kernel->tpe_gradient_pair(face1, mnp2, v));
                    addToRow(gradients, indices[v], kernel->tpe_gradient_pair(mnp2, face1, v));
                }
            }
            else
            {
                // Otherwise we continue recursively traversing the tree
                for (size_t i = 0; i < node->children.size(); i++)
                {
                    if (node->children[i])
                    {
                        accumulateTPEGradient(gradients, node->children[i], face1, indices);
                    }
                }
            }
        }
    }

    void BarnesHutTPEnergy6D::Update()
    {
        if (root)
        {
            delete root;
        }
        root = Create6DBVHFromMeshFaces(kernel->mesh, kernel->geom, theta);
    }

    MeshPtr BarnesHutTPEnergy6D::GetMesh()
    {
        return kernel->mesh;
    }

    GeomPtr BarnesHutTPEnergy6D::GetGeom()
    {
        return kernel->geom;
    }

    Vector2 BarnesHutTPEnergy6D::GetExponents()
    {
        return Vector2{kernel->alpha, kernel->beta};
    }

} // namespace rsurfaces
