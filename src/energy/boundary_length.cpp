#include "energy/boundary_length.h"
#include "matrix_utils.h"

namespace rsurfaces
{
    BoundaryLengthPenalty::BoundaryLengthPenalty(MeshPtr mesh_, GeomPtr geom_, double weight_, double targetValue_)
    {
        weight = weight_;
        mesh = mesh_;
        geom = geom_;
        targetValue = targetValue_;

        if (mesh->nBoundaryLoops() == 0)
        {
            throw std::runtime_error("Boundary length penalty was specified, but mesh has no boundary loops.");
        }
    }

    double BoundaryLengthPenalty::currentLengthDeviation()
    {
        double sum = 0;
        // The energy is just the total length of the boundary loops.
        for (surface::BoundaryLoop loop : mesh->boundaryLoops())
        {
            for (GCEdge e : loop.adjacentEdges())
            {
                sum += geom->edgeLength(e);
            }
        }

        return (sum - targetValue);
    }


    double BoundaryLengthPenalty::Value()
    {
        double deviation = currentLengthDeviation();
        return weight * deviation * deviation;
    }

    void BoundaryLengthPenalty::Differential(Eigen::MatrixXd &output)
    {
        double deviation = currentLengthDeviation();
        std::cout << "  * Boundary length = " << (targetValue + deviation) << std::endl;

        VertexIndices inds = mesh->getVertexIndices();

        for (surface::BoundaryLoop loop : mesh->boundaryLoops())
        {
            for (GCEdge e : loop.adjacentEdges())
            {
                GCVertex v1 = e.firstVertex();
                GCVertex v2 = e.secondVertex();

                Vector3 awayFromFirst = (geom->inputVertexPositions[v2] - geom->inputVertexPositions[v1]).normalize();
                Vector3 deriv = 2 * deviation * awayFromFirst;

                // Gradient wants to make the edge longer at both vertices.
                MatrixUtils::addToRow(output, inds[v2], weight * deriv);
                MatrixUtils::addToRow(output, inds[v1], -weight * deriv);
            }
        }
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 BoundaryLengthPenalty::GetExponents()
    {
        return Vector2{2, 0};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree* BoundaryLengthPenalty::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double BoundaryLengthPenalty::GetTheta()
    {
        return 0;
    }
}