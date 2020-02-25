#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
class FractionalLaplacian
{
    Vector3 HatGradientOnFace(const GeomPtr &geom, GCFace face, GCVertex vertex);
    void AddTriPairValues(const GeomPtr &geom, GCFace S, GCFace T, double s_pow,
                         surface::VertexData<size_t> &indices, Eigen::MatrixXd &A);

    void FillMatrix(const MeshPtr &mesh, const GeomPtr &geom, double s_pow, Eigen::MatrixXd &A);
};

} // namespace rsurfaces
