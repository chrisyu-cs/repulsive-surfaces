#include "all_pairs_tpe.h"

#include <unordered_set>

namespace rsurfaces
{

AllPairsTPEnergy::AllPairsTPEnergy(TPEKernel *kernel_)
{
    kernel = kernel_;
}

double AllPairsTPEnergy::Value()
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

void AllPairsTPEnergy::Differential(Eigen::MatrixXd &output)
{
    surface::VertexData<size_t> indices = kernel->mesh->getVertexIndices();

    for (GCFace f1 : kernel->mesh->faces())
    {
        for (GCFace f2 : kernel->mesh->faces())
        {
            if (f1 == f2)
                continue;
            // Find the set of vertices that are on the boundary of either
            // triangle, without duplicates
            std::unordered_set<size_t> vertInds;
            std::vector<GCVertex> verts;
            for (GCVertex v : f1.adjacentVertices())
            {
                if (vertInds.count(indices[v]) == 0)
                {
                    verts.push_back(v);
                    vertInds.insert(indices[v]);
                }
            }
            for (GCVertex v : f2.adjacentVertices())
            {
                if (vertInds.count(indices[v]) == 0)
                {
                    verts.push_back(v);
                    vertInds.insert(indices[v]);
                }
            }

            for (GCVertex v : verts)
            {
                Vector3 deriv1 = kernel->tpe_gradient_pair(f1, f2, v);
                Vector3 deriv2 = kernel->tpe_gradient_pair(f2, f1, v);
                Vector3 sum = deriv1 + deriv2;
                int r = indices[v];
                output(r, 0) = sum.x;
                output(r, 1) = sum.y;
                output(r, 2) = sum.z;
            }
        }
    }
}

} // namespace rsurfaces
