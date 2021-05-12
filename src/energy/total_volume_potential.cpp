#include "energy/total_volume_potential.h"
#include "matrix_utils.h"
#include "surface_derivatives.h"

namespace rsurfaces
{
    TotalVolumePotential::TotalVolumePotential(MeshPtr mesh_, GeomPtr geom_, double weight_)
    {
        mesh = mesh_;
        geom = geom_;
        weight = weight_;
    }

    // Returns the current value of the energy.
    double TotalVolumePotential::Value()
    {
        double v = totalVolume(geom, mesh);
        return weight * v * v;
    }

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void TotalVolumePotential::Differential(Eigen::MatrixXd &output)
    {
        double v = totalVolume(geom, mesh);
        VertexIndices inds = mesh->getVertexIndices();
        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            GCVertex v_i = mesh->vertex(i);
            Vector3 deriv_v =  areaWeightedNormal(geom, v_i);
            // Derivative of V^2 = V * (deriv V) = V * (area normal)
            MatrixUtils::addToRow(output, inds[v_i], weight * v * deriv_v);
        }
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 TotalVolumePotential::GetExponents()
    {
        return Vector2{1, 0};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *TotalVolumePotential::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double TotalVolumePotential::GetTheta()
    {
        return 0;
    }

} // namespace rsurfaces