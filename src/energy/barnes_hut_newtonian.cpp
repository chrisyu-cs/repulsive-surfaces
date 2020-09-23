#include "energy/barnes_hut_newtonian.h"
#include "helpers.h"
#include "surface_derivatives.h"
#include "matrix_utils.h"

namespace rsurfaces
{

    BarnesHutNewtonian::BarnesHutNewtonian(TPEKernel *kernel_, double theta_)
    {
        kernel = kernel_;
        theta = theta_;
        root = 0;
        flattened = 0;
    }

    BarnesHutNewtonian::~BarnesHutNewtonian()
    {
        if (root)
        {
            delete root;
        }
    }

    double BarnesHutNewtonian::Value()
    {
        return 0;
    }

    double BarnesHutNewtonian::computeEnergyOfFace(GCFace face, BVHNode6D *bvhRoot)
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
        else if (bvhRoot->isAdmissibleFrom(bcenter, theta))
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

    void BarnesHutNewtonian::Differential(Eigen::MatrixXd &output)
    {
        if (!root)
        {
            std::cerr << "BVH for BarnesHutNewtonian was not initialized. Call Update() first." << std::endl;
            std::exit(1);
        }

        VertexIndices indices = kernel->mesh->getVertexIndices();
        output.setZero();
        Eigen::MatrixXd partialOutput = output;
        #pragma omp parallel firstprivate(partialOutput) shared(root, output)
        {
            #pragma omp for
            for (size_t i = 0; i < kernel->mesh->nFaces(); i++)
            {
                GCFace f = kernel->mesh->face(i);
                accumulateForce(partialOutput, root, f, indices);
            }

            #pragma omp critical
            {
                output += partialOutput;
            }
        }
    }

    // Add derivatives of all energy terms of the form (f1, _) or (_, f1)
    // with respect to the neighbor vertices of f1.
    void BarnesHutNewtonian::accumulateForce(Eigen::MatrixXd &gradients, BVHNode6D *node, GCFace face1,
                                              surface::VertexData<size_t> &indices)
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

            Vector3 force = kernel->newton_pair(face1, face2);

            for (GCVertex v : face1.adjacentVertices())
            {
                size_t r = indices[v];
                MatrixUtils::addToRow(gradients, r, force / 3);
            }
        }
        else
        {
            Vector3 f1_center = faceBarycenter(kernel->geom, face1);
            if (node->isAdmissibleFrom(f1_center, theta))
            {
                Vector3 normal = node->averageNormal;
                // This cell is far enough away that we can treat it as a single body
                MassNormalPoint mnp2 = node->GetMassNormalPoint();
                Vector3 force = kernel->newton_pair(face1, mnp2);

                // Differentiate both terms for all neighbors
                for (GCVertex v : face1.adjacentVertices())
                {
                    // Derivatives of both foward and reverse terms
                    MatrixUtils::addToRow(gradients, indices[v], force / 3);
                }
            }
            else
            {
                // Otherwise we continue recursively traversing the tree
                for (size_t i = 0; i < BVH_N_CHILDREN; i++)
                {
                    accumulateForce(gradients, node->children[i], face1, indices);
                }
            }
        }
    }

    void BarnesHutNewtonian::Update()
    {
        if (root)
        {
            delete root;
        }
        if (flattened)
        {
            delete flattened;
        }

        root = Create6DBVHFromMeshFaces(kernel->mesh, kernel->geom);
        flattened = new BVHFlattened(root);
    }

    MeshPtr BarnesHutNewtonian::GetMesh()
    {
        return kernel->mesh;
    }

    GeomPtr BarnesHutNewtonian::GetGeom()
    {
        return kernel->geom;
    }

    Vector2 BarnesHutNewtonian::GetExponents()
    {
        return Vector2{kernel->alpha, kernel->beta};
    }

    BVHNode6D *BarnesHutNewtonian::GetBVH()
    {
        return root;
    }

    double BarnesHutNewtonian::GetTheta()
    {
        return theta;
    }

} // namespace rsurfaces
