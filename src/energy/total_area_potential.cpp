#include "energy/total_area_potential.h"
#include "matrix_utils.h"
#include "surface_derivatives.h"

namespace rsurfaces
{
    TotalAreaPotential::TotalAreaPotential(MeshPtr mesh_, GeomPtr geom_, double weight_)
    {
        mesh = mesh_;
        geom = geom_;
        weight = weight_;
    }

    // Returns the current value of the energy.
    double TotalAreaPotential::Value()
    {
        double a = totalArea(geom, mesh);
        return weight * a * a;
    }

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void TotalAreaPotential::Differential(Eigen::MatrixXd &output)
    {
        VertexIndices inds = mesh->getVertexIndices();
        double a = totalArea(geom, mesh);

        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            GCVertex v_i = mesh->vertex(i);
            Vector3 sumDerivs{0, 0, 0};

            // Each vertex produces a derivative wrt its surrounding faces
            for (GCFace f : v_i.adjacentFaces())
            {
                sumDerivs += SurfaceDerivs::triangleAreaWrtVertex(geom, f, v_i);
            }
            MatrixUtils::addToRow(output, inds[v_i], weight * a * sumDerivs);
        }
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 TotalAreaPotential::GetExponents()
    {
        return Vector2{1, 0};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *TotalAreaPotential::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double TotalAreaPotential::GetTheta()
    {
        return 0;
    }

} // namespace rsurfaces
