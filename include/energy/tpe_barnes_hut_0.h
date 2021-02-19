#pragma once

#include "rsurface_types.h"
#include "surface_energy.h"
#include "helpers.h"
#include "block_cluster_tree2_types.h"
#include "block_cluster_tree2.h"
#include "derivative_assembler.h"

namespace rsurfaces
{

    // 0-th order multipole approximation of tangent-point energy
    class TPEnergyBarnesHut0 : public SurfaceEnergy
    {
    public:
        ~TPEnergyBarnesHut0(){};
        
        TPEnergyBarnesHut0( MeshPtr mesh_, GeomPtr geom_, BlockClusterTree2 * bct_ )
        {
            mesh = mesh_;
            geom = geom_;
            bct = bct_;
        }
        
        // Returns the current value of the energy.
        virtual double Value();

        // Returns the current differential of the energy, stored in the given
        // V x 3 matrix, where each row holds the differential (a 3-vector) with
        // respect to the corresponding vertex.
        virtual void Differential(Eigen::MatrixXd &output);

        // Update the energy to reflect the current state of the mesh. This could
        // involve building a new BVH for Barnes-Hut energies, for instance.
        virtual void Update();

        // Get the mesh associated with this energy.
        virtual MeshPtr GetMesh();

        // Get the geometry associated with this geometry.
        virtual GeomPtr GetGeom();

        // Get the exponents of this energy; only applies to tangent-point energies.
        virtual Vector2 GetExponents();

        // Get a pointer to the current BVH for this energy.
        // Return 0 if the energy doesn't use a BVH.
        virtual BVHNode6D *GetBVH();

        // Return the separation parameter for this energy.
        // Return 0 if this energy doesn't do hierarchical approximation.
        virtual double GetTheta();
        
        BlockClusterTree2 * GetBCT(){ return bct; };
        
    private:
        
        MeshPtr mesh = nullptr;
        GeomPtr geom = nullptr;
        BlockClusterTree2 * bct = nullptr;
        
    }; // TPEnergyBarnesHut0

} // namespace rsurfaces
