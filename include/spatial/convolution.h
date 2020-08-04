#pragma once

#include "bvh_6d.h"
#include "matrix_utils.h"

namespace rsurfaces
{
    inline void FixBarycenter(MeshPtr &mesh, GeomPtr &geom, VertexIndices &indices, Eigen::MatrixXd &data)
    {
        double sumArea = 0;
        Vector3 average{0, 0, 0};
        for (GCVertex v1 : mesh->vertices())
        {
            double area = geom->vertexDualAreas[v1];
            average += GetRow(data, indices[v1]) * area;
            sumArea += area;
        }
        average /= sumArea;
        Eigen::RowVector3d rowAvg{average.x, average.y, average.z};

        for (int i = 0; i < data.rows(); i++)
        {
            data.row(i) -= rowAvg;
        }
    }

    template <typename Kernel>
    void ConvolveExact(MeshPtr &mesh, GeomPtr &geom, Kernel &ker, Eigen::MatrixXd &data, Eigen::MatrixXd &output)
    {
        VertexIndices indices = mesh->getVertexIndices();

        for (GCVertex v1 : mesh->vertices())
        {
            Vector3 result1{0, 0, 0};
            Vector3 p1 = geom->inputVertexPositions[v1];
            double sumWeight = 0;

            for (GCVertex v2 : mesh->vertices())
            {
                Vector3 val2 = GetRow(data, indices[v2]);
                Vector3 p2 = geom->inputVertexPositions[v2];
                double wt =  ker.Coefficient(p1, p2) * geom->vertexDualAreas[v2];
                result1 += val2 * wt;
                sumWeight += wt;
            }
            MatrixUtils::SetRowFromVector3(output, indices[v1], result1 / sumWeight);
        }
        // FixBarycenter(mesh, geom, indices, output);
    }

} // namespace rsurfaces
