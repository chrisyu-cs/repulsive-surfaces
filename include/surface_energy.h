#pragma once

#include "rsurface_types.h"
#include "optimized_cluster_tree.h"
#include "metric_term.h"

namespace rsurfaces
{
    class SurfaceEnergy
    {
    public:
        virtual ~SurfaceEnergy() {}

        // If this energy has constant values, resets them based on
        // the current mesh configuration.
        virtual void ResetTargets() {}

        // Returns the current value of the energy.
        virtual double Value() = 0;
        
        // Returns the current differential of the energy, stored in the given
        // V x 3 matrix, where each row holds the differential (a 3-vector) with
        // respect to the corresponding vertex.
        virtual void Differential(Eigen::MatrixXd &output) = 0;
        
        // Update the energy to reflect the current state of the mesh. This could
        // involve building a new BVH for Barnes-Hut energies, for instance.
        virtual void Update() {}
        
        // Get the mesh associated with this energy.
        virtual MeshPtr GetMesh()
        {
            return mesh;
        }
        
        // Get the geometry associated with this geometry.
        virtual GeomPtr GetGeom()
        {
            return geom;
        }
        
        // Get the exponents of this energy; only applies to tangent-point energies.
        virtual Vector2 GetExponents() = 0;
        
        // Get a pointer to the current BVH for this energy.
        // Return 0 if the energy doesn't use a BVH.
        virtual OptimizedClusterTree *GetBVH()
        {
            return 0;
        }
        
        // Return the separation parameter for this energy.
        // Return 0 if this energy doesn't do hierarchical approximation.
        virtual double GetTheta()
        {
            return 0;
        }
        
        virtual double GetWeight()
        {
            return weight;
        }

        // Add the metric term associated with this energy term, if applicable.
        // If no special metric term is needed, does nothing.
        // For Hs and obstacle terms, this also does nothing -- the metric is
        // specially constructed elsewhere.
        virtual void AddMetricTerm(std::vector<MetricTerm*> &terms) {}

    protected:
        MeshPtr mesh = 0;
        GeomPtr geom = 0;
        double weight = 1.;
    };
} // namespace rsurfaces
