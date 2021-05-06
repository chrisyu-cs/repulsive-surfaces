#pragma once

#include "rsurface_types.h"
#include "surface_energy.h"
#include "helpers.h"
#include "optimized_bct_types.h"
#include "derivative_assembler.h"

namespace rsurfaces
{
    class WillmoreEnergy : public SurfaceEnergy
    {
    public:
        ~WillmoreEnergy(){};
        
        WillmoreEnergy( MeshPtr mesh_, GeomPtr geom_, double wt = 1. )
        {
            mesh = mesh_;
            geom = geom_;
            weight = wt;
        }
        
        // Returns the current value of the energy.
        virtual double Value();

        // Returns the current differential of the energy, stored in the given
        // V x 3 matrix, where each row holds the differential (a 3-vector) with
        // respect to the corresponding vertex.
        virtual void Differential(Eigen::MatrixXd &output);

        // Get the exponents of this energy; only applies to tangent-point energies.
        virtual Vector2 GetExponents();

        // Willmore energy should add a bi-Laplacian term.
        virtual void AddMetricTerm(std::vector<MetricTerm*> &terms);

        void requireMeanCurvatureVectors();
        
    private:
        double weight = 1.;
        Eigen::MatrixXd H;
        Eigen::VectorXd H_squared;
    }; // WillmoreEnergy
} // namespace rsurfaces
