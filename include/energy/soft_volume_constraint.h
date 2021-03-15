#pragma once

#include "rsurface_types.h"
#include "surface_energy.h"

namespace rsurfaces
{
    class SoftVolumeConstraint : public SurfaceEnergy
    {
    public:
        SoftVolumeConstraint(MeshPtr mesh_, GeomPtr geom_, double weight_);

        virtual ~SoftVolumeConstraint() {}

        // Returns the current value of the energy.
        virtual double Value();

        // Returns the current differential of the energy, stored in the given
        // V x 3 matrix, where each row holds the differential (a 3-vector) with
        // respect to the corresponding vertex.
        virtual void Differential(Eigen::MatrixXd &output);

        // Get the mesh associated with this energy.
        virtual MeshPtr GetMesh();

        // Get the geometry associated with this geometry.
        virtual GeomPtr GetGeom();

        // Get the exponents of this energy; only applies to tangent-point energies.
        virtual Vector2 GetExponents();

        // Get a pointer to the current BVH for this energy.
        // Return 0 if the energy doesn't use a BVH.
        virtual OptimizedClusterTree *GetBVH();

        // Return the separation parameter for this energy.
        // Return 0 if this energy doesn't do hierarchical approximation.
        virtual double GetTheta();

    private: 
        MeshPtr mesh;
        GeomPtr geom;
        double initialVolume;
        double weight;
    };
}