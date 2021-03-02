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
        return weight * totalVolume(geom, mesh);
    }

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void TotalVolumePotential::Differential(Eigen::MatrixXd &output)
    {
        VertexIndices inds = mesh->getVertexIndices();
        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            GCVertex v_i = mesh->vertex(i);
            // Derivative of local volume is just the area weighted normal
            Vector3 deriv_v = areaWeightedNormal(geom, v_i);
            MatrixUtils::addToRow(output, inds[v_i], weight * deriv_v);
        }
    }

    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void TotalVolumePotential::Update()
    {
        // Nothing needs to be done
    }

    // Get the mesh associated with this energy.
    MeshPtr TotalVolumePotential::GetMesh()
    {
        return mesh;
    }

    // Get the geometry associated with this geometry.
    GeomPtr TotalVolumePotential::GetGeom()
    {
        return geom;
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