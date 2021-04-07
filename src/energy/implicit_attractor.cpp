#include "energy/implicit_attractor.h"
#include "matrix_utils.h"

namespace rsurfaces
{

    ImplicitAttractor::ImplicitAttractor(MeshPtr mesh_, GeomPtr geom_, std::unique_ptr<ImplicitSurface> surface_, UVDataPtr uvs_, double power_, double w)
        : surface(std::move(surface_))
    {
        power = power_;
        weight = w;
        mesh = mesh_;
        geom = geom_;
        uvs = uvs_;

        if (uvs)
        {
            std::cout << "Using UVs to determine which vertices get attracted" << std::endl;
        }
        else
        {
            std::cout << "No UVs; attractor will attract all vertices" << std::endl;
        }
    }

    double ImplicitAttractor::Value()
    {
        double sum = 0;
        for (GCVertex v : mesh->vertices())
        {
            if (shouldAttract(v))
            {
                double signDist = surface->SignedDistance(geom->inputVertexPositions[v]);
                sum += pow(signDist, power);
            }
        }
        return weight * sum;
    }

    void ImplicitAttractor::Differential(Eigen::MatrixXd &output)
    {
        VertexIndices inds = mesh->getVertexIndices();

        for (GCVertex v : mesh->vertices())
        {
            if (shouldAttract(v))
            {
                Vector3 gradDist = surface->GradientOfDistance(geom->inputVertexPositions[v]);
                double signDist = surface->SignedDistance(geom->inputVertexPositions[v]);

                // d/dx D^2 = 2 * D * (d/dx D)
                Vector3 gradE = power * pow(signDist, power - 1) * gradDist;
                MatrixUtils::addToRow(output, inds[v], weight * gradE);
            }
        }
    }

    Vector2 ImplicitAttractor::GetExponents()
    {
        return Vector2{0, 2};
    }

    OptimizedClusterTree *ImplicitAttractor::GetBVH()
    {
        return 0;
    }

    double ImplicitAttractor::GetTheta()
    {
        return 0;
    }
} // namespace rsurfaces