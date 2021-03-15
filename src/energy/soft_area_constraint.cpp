#include "energy/soft_area_constraint.h"
#include "matrix_utils.h"
#include "surface_derivatives.h"

namespace rsurfaces
{
    SoftAreaConstraint::SoftAreaConstraint(MeshPtr mesh_, GeomPtr geom_, double weight_)
    {
        mesh = mesh_;
        geom = geom_;
        weight = weight_;
        initialArea = totalArea(geom, mesh);
    }

    inline double areaDeviation(MeshPtr mesh, GeomPtr geom, double initialArea)
    {
        return (totalArea(geom, mesh) - initialArea) / initialArea;
    }

    // Returns the current value of the energy.
    double SoftAreaConstraint::Value()
    {
        double areaDev = areaDeviation(mesh, geom, initialArea);
        return weight * (areaDev * areaDev);
    }

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void SoftAreaConstraint::Differential(Eigen::MatrixXd &output)
    {
        double areaDev = areaDeviation(mesh, geom, initialArea);

        VertexIndices inds = mesh->getVertexIndices();
        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            GCVertex v_i = mesh->vertex(i);
            Vector3 sumDerivs{0, 0, 0};

            // Each vertex produces a derivative wrt its surrounding faces
            for (GCFace f : v_i.adjacentFaces())
            {
                if (f.isBoundaryLoop())
                    continue;
                sumDerivs += SurfaceDerivs::triangleAreaWrtVertex(geom, f, v_i);
            }
            // Differential of A^2 = 2 A (dA/dx)
            sumDerivs = 2 * areaDev * sumDerivs / initialArea;
            MatrixUtils::addToRow(output, inds[v_i], weight * geom->vertexDualAreas[v_i] * sumDerivs);
        }
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 SoftAreaConstraint::GetExponents()
    {
        return Vector2{1, 0};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree *SoftAreaConstraint::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double SoftAreaConstraint::GetTheta()
    {
        return 0;
    }

} // namespace rsurfaces