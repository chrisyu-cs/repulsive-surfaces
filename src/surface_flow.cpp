#include "surface_flow.h"
#include "fractional_laplacian.h"
#include "helpers.h"

namespace rsurfaces
{

SurfaceFlow::SurfaceFlow(SurfaceEnergy *energy_)
{
    energy = energy_;
    mesh = energy->GetMesh();
    geom = energy->GetGeom();
}

void SurfaceFlow::StepNaive(double t)
{
    Eigen::MatrixXd gradient;
    gradient.setZero(mesh->nVertices(), 3);

    energy->Differential(gradient);

    surface::VertexData<size_t> indices = mesh->getVertexIndices();

    for (GCVertex v : mesh->vertices()) {
        Vector3 grad_v = GetRow(gradient, indices[v]);
        geom->inputVertexPositions[v] -= grad_v * t;
    }
}

} // namespace rsurfaces