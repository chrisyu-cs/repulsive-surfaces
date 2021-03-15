#pragma once

#include "rsurface_types.h"
#include "surface_energy.h"
#include "helpers.h"
#include "bct_constructors.h"
#include "optimized_bct_types.h"
#include "optimized_cluster_tree.h"
#include "derivative_assembler.h"

namespace rsurfaces
{

    // 0-th order multipole approximation of tangent-point energy
    class TPObstacleBarnesHut0 : public SurfaceEnergy
    {
    public:
        TPObstacleBarnesHut0(MeshPtr mesh_, GeomPtr geom_, SurfaceEnergy *bvhSharedFrom_, MeshPtr &obsMesh, GeomPtr &obsGeom,
                             mreal alpha_, mreal beta_, mreal theta_, mreal weight_ = 1.)
        {
            mesh = mesh_;
            geom = geom_;
            bvh = 0;
            bvhSharedFrom = bvhSharedFrom_;
            o_bvh = CreateOptimizedBVH(obsMesh, obsGeom);
            alpha = alpha_;
            beta = beta_;
            theta = theta_;
            weight = weight_;

            mreal intpart;
            use_int = (std::modf(alpha, &intpart) == 0.0) && (std::modf(beta / 2, &intpart) == 0.0);
            
            Update();
        }

        ~TPObstacleBarnesHut0()
        {
            if (o_bvh)
            {
                delete o_bvh;
            }
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
        virtual OptimizedClusterTree *GetBVH();

        // Return the separation parameter for this energy.
        // Return 0 if this energy doesn't do hierarchical approximation.
        virtual double GetTheta();

        bool use_int = false;

    private:
        MeshPtr mesh = nullptr;
        GeomPtr geom = nullptr;
        mreal alpha = 6.;
        mreal beta = 12.;
        mreal weight = 1.;
        mreal theta = 0.5;

        SurfaceEnergy * bvhSharedFrom;
        OptimizedClusterTree * bvh = nullptr;
        OptimizedClusterTree * o_bvh = nullptr;

        template <typename T1, typename T2>
        mreal Energy(T1 alpha, T2 betahalf);

        template <typename T1, typename T2>
        mreal DEnergy(T1 alpha, T2 betahalf);

    }; // TPEnergyBarnesHut0

} // namespace rsurfaces
