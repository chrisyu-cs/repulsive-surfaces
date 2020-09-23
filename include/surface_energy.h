#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
    class SurfaceEnergy
    {
    public:
        virtual ~SurfaceEnergy() {}
        // Returns the current value of the energy.
        virtual double Value() = 0;
        // Returns the current differential of the energy, stored in the given
        // V x 3 matrix, where each row holds the differential (a 3-vector) with
        // respect to the corresponding vertex.
        virtual void Differential(Eigen::MatrixXd &output) = 0;
        // Update the energy to reflect the current state of the mesh. This could
        // involve building a new BVH for Barnes-Hut energies, for instance.
        virtual void Update() = 0;
        // Get the mesh associated with this energy.
        virtual MeshPtr GetMesh() = 0;
        // Get the geometry associated with this geometry.
        virtual GeomPtr GetGeom() = 0;
        // Get the exponents of this energy; only applies to tangent-point energies.
        virtual Vector2 GetExponents() = 0;
        // Get a pointer to the current BVH for this energy.
        // Return 0 if the energy doesn't use a BVH.
        virtual BVHNode6D *GetBVH() = 0;
        // Return the separation parameter for this energy.
        // Return 0 if this energy doesn't do hierarchical approximation.
        virtual double GetTheta() = 0;
    };
} // namespace rsurfaces