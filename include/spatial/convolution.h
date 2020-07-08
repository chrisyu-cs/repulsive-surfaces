#pragma once

#include "bvh_6d.h"
#include "matrix_utils.h"

namespace rsurfaces
{
    template <typename Kernel>
    Vector3 ConvolveAtPointBH(MeshPtr &mesh, GeomPtr &geom, Kernel &ker, BVHNode6D *node, Vector3 center)
    {
        if (node->nodeType == BVHNodeType::Empty)
        {
            return Vector3{0, 0, 0};
        }
        else if (node->nodeType == BVHNodeType::Leaf)
        {
            return node->customData * ker.Coefficient(center, node->centerOfMass) * node->totalMass;
        }
        else
        {
            if (node->isAdmissibleFrom(center))
            {
                return node->customData * ker.Coefficient(center, node->centerOfMass) * node->totalMass;
            }
            else
            {
                Vector3 sum{0, 0, 0};
                // Otherwise we continue recursively traversing the tree
                for (size_t i = 0; i < node->children.size(); i++)
                {
                    if (node->children[i])
                    {
                        sum += ConvolveAtPointBH(mesh, geom, ker, node->children[i], center);
                    }
                }
                return sum;
            }
        }
    }

    template <typename Kernel>
    void ConvolveBH(MeshPtr &mesh, GeomPtr &geom, Kernel &ker, BVHNode6D *bvh, Eigen::MatrixXd &data, Eigen::MatrixXd &output)
    {
        VertexIndices indices = mesh->getVertexIndices();
        bvh->propagateCustomData(data);

        for (GCVertex vert : mesh->vertices())
        {
            Vector3 pos = geom->inputVertexPositions[vert];
            Vector3 result = ConvolveAtPointBH(mesh, geom, ker, bvh, pos);
            MatrixUtils::SetRowFromVector3(output, indices[vert], result);
        }
    }

    template <typename Kernel>
    void ConvolveExact(MeshPtr &mesh, GeomPtr &geom, Kernel &ker, Eigen::MatrixXd &data, Eigen::MatrixXd &output)
    {
        VertexIndices indices = mesh->getVertexIndices();

        for (GCVertex v1 : mesh->vertices())
        {
            Vector3 result1{0, 0, 0};
            for (GCVertex v2 : mesh->vertices())
            {
                Vector3 val2 = GetRow(data, indices[v2]);
                double area2 = geom->vertexDualAreas[v2];
                Vector3 p1 = geom->inputVertexPositions[v1];
                Vector3 p2 = geom->inputVertexPositions[v2];
                Vector3 contrib = val2 * ker.Coefficient(p1, p2) * area2;
                result1 += contrib;
            }
            MatrixUtils::SetRowFromVector3(output, indices[v1], result1);
        }
    }

} // namespace rsurfaces
