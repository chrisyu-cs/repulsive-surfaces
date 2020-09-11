#include "energy/all_pairs_with_bvh.h"
#include "helpers.h"

#include <unordered_set>

namespace rsurfaces
{

    AllPairsWithBVH::AllPairsWithBVH(TPEKernel *kernel_, double theta_)
    {
        kernel = kernel_;
        theta = theta_;
        root = 0;
    }

    double AllPairsWithBVH::Value()
    {
        double total = 0;
        for (GCFace f1 : kernel->mesh->faces())
        {
            for (GCFace f2 : kernel->mesh->faces())
            {
                if (f1 == f2)
                    continue;
                total += kernel->tpe_pair(f1, f2);
            }
        }
        return total;
    }

    void AllPairsWithBVH::Differential(Eigen::MatrixXd &output)
    {
        output.setZero();
        surface::VertexData<size_t> indices = kernel->mesh->getVertexIndices();
        for (GCFace f1 : kernel->mesh->faces())
        {
            for (GCFace f2 : kernel->mesh->faces())
            {
                if (f1 == f2)
                    continue;
                // Find the set of vertices that are on the boundary of either
                // triangle, without duplicates
                std::vector<GCVertex> verts;
                GetVerticesWithoutDuplicates(f1, f2, verts);

                for (GCVertex &v : verts)
                {
                    Vector3 deriv1 = kernel->tpe_gradient_pair(f1, f2, v);
                    Vector3 sum = deriv1;
                    size_t r = indices[v];

                    output(r, 0) += sum.x;
                    output(r, 1) += sum.y;
                    output(r, 2) += sum.z;
                }
            }
        }
    }

    void AllPairsWithBVH::Update() {
        if (root)
        {
            delete root;
        }
        root = Create6DBVHFromMeshFaces(kernel->mesh, kernel->geom, theta);
    }

    MeshPtr AllPairsWithBVH::GetMesh()
    {
        return kernel->mesh;
    }

    GeomPtr AllPairsWithBVH::GetGeom()
    {
        return kernel->geom;
    }

    Vector2 AllPairsWithBVH::GetExponents()
    {
        return Vector2{kernel->alpha, kernel->beta};
    }

    BVHNode6D* AllPairsWithBVH::GetBVH() {
        return root;
    }

    double AllPairsWithBVH::GetTheta()
    {
        return theta;
    }

} // namespace rsurfaces
