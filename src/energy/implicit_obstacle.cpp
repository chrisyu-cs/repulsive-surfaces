#include "energy/implicit_obstacle.h"
#include "matrix_utils.h"

namespace rsurfaces
{

    ImplicitObstacle::ImplicitObstacle(MeshPtr mesh_, GeomPtr geom_, std::unique_ptr<ImplicitSurface> surface_, double power_, double w)
        : surface(std::move(surface_))
    {
        power = power_;
        weight = w;
        mesh = mesh_;
        geom = geom_;
    }

    double ImplicitObstacle::Value()
    {
        double sum = 0;
        for (GCVertex v : mesh->vertices())
        {
            double signDist = surface->SignedDistance(geom->inputVertexPositions[v]);
            sum += 1.0 / pow(signDist, power);
        }
        return weight * sum;
    }

    void ImplicitObstacle::Differential(Eigen::MatrixXd &output)
    {
        VertexIndices inds = mesh->getVertexIndices();

        for (GCVertex v : mesh->vertices())
        {
            Vector3 gradDist = surface->GradientOfDistance(geom->inputVertexPositions[v]);
            double signDist = surface->SignedDistance(geom->inputVertexPositions[v]);

            // d/dx (1 / D^x) = -x * (1 / D^(x+1)) * (d/dx D)
            double coeff = -power * (1.0 / pow(signDist, power + 1));
            Vector3 gradE = coeff * gradDist;
            MatrixUtils::addToRow(output, inds[v], weight * gradE);
        }
    }

    Vector2 ImplicitObstacle::GetExponents()
    {
        return Vector2{0, 2};
    }

    OptimizedClusterTree *ImplicitObstacle::GetBVH()
    {
        return 0;
    }

    double ImplicitObstacle::GetTheta()
    {
        return 0;
    }
} // namespace rsurfaces