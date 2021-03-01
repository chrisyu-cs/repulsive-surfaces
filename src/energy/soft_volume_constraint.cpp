#include "energy/soft_volume_constraint.h"
#include "matrix_utils.h"
#include "surface_derivatives.h"

namespace rsurfaces
{
    SoftVolumeConstraint::SoftVolumeConstraint(MeshPtr mesh_, GeomPtr geom_, double weight_)
    {
        mesh = mesh_;
        geom = geom_;
        weight = weight_;
        initialVolume = totalVolume(geom, mesh);
    }

    inline double volumeDeviation(MeshPtr mesh, GeomPtr geom, double initialValue)
    {
        return (totalVolume(geom, mesh) - initialValue) / initialValue;
    }

    // Returns the current value of the energy.
    double SoftVolumeConstraint::Value()
    {
        double volDev = volumeDeviation(mesh, geom, initialVolume);
        return weight * (volDev * volDev);
    }

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void SoftVolumeConstraint::Differential(Eigen::MatrixXd &output)
    {
        double volDev = volumeDeviation(mesh, geom, initialVolume);

        VertexIndices inds = mesh->getVertexIndices();
        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            GCVertex v_i = mesh->vertex(i);
            // Derivative of local volume is just the area weighted normal
            Vector3 deriv_v = areaWeightedNormal(geom, v_i);
            // Derivative of V^2 = 2 V (dV/dx)
            deriv_v = 2 * volDev * deriv_v / initialVolume;

            MatrixUtils::addToRow(output, inds[v_i], weight * deriv_v);
        }
    }

    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void SoftVolumeConstraint::Update()
    {
        // Nothing needs to be done
    }

    // Get the mesh associated with this energy.
    MeshPtr SoftVolumeConstraint::GetMesh()
    {
        return mesh;
    }

    // Get the geometry associated with this geometry.
    GeomPtr SoftVolumeConstraint::GetGeom()
    {
        return geom;
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 SoftVolumeConstraint::GetExponents()
    {
        return Vector2{1, 0};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *SoftVolumeConstraint::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double SoftVolumeConstraint::GetTheta()
    {
        return 0;
    }

} // namespace rsurfaces