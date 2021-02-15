#include "energy/implicit_attractor.h"
#include "matrix_utils.h"

namespace rsurfaces
{

    ImplicitAttractor::ImplicitAttractor(MeshPtr mesh_, GeomPtr geom_, std::unique_ptr<ImplicitSurface> surface_, double w)
        : surface(std::move(surface_))
    {
        weight = w;
        mesh = mesh_;
        geom = geom_;
    }

    double ImplicitAttractor::Value()
    {
        double sum = 0;
        for (GCVertex v : mesh->vertices())
        {
            double signDist = surface->SignedDistance(geom->inputVertexPositions[v]);
            sum += (signDist * signDist);
        }
        return weight * sum;
    }

    void ImplicitAttractor::Differential(Eigen::MatrixXd &output)
    {
        VertexIndices inds = mesh->getVertexIndices();

        for (GCVertex v : mesh->vertices())
        {
            Vector3 gradDist = surface->GradientOfDistance(geom->inputVertexPositions[v]);
            double signDist = surface->SignedDistance(geom->inputVertexPositions[v]);

            // d/dx D^2 = 2 * D * (d/dx D)
            Vector3 gradE = 2 * signDist * gradDist;
            MatrixUtils::addToRow(output, inds[v], weight * gradE);
        }
    }

    void ImplicitAttractor::Update()
    {
        // Nothing to do
    }

    MeshPtr ImplicitAttractor::GetMesh()
    {
        return mesh;
    }

    GeomPtr ImplicitAttractor::GetGeom()
    {
        return geom;
    }

    Vector2 ImplicitAttractor::GetExponents()
    {
        return Vector2{0, 2};
    }

    BVHNode6D *ImplicitAttractor::GetBVH()
    {
        return 0;
    }

    double ImplicitAttractor::GetTheta()
    {
        return 0;
    }
} // namespace rsurfaces