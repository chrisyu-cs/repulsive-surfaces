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

    // Returns the current value of the energy.
    double SoftAreaConstraint::Value()
    {
        double areaDev = totalArea(geom, mesh) - initialArea;
        return weight * (areaDev * areaDev);
    }

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void SoftAreaConstraint::Differential(Eigen::MatrixXd &output)
    {
        double currentValue = Value();

        VertexIndices inds = mesh->getVertexIndices();
        #pragma omp parallel shared(output)
        {
            #pragma omp parallel for
            for (size_t i = 0; i < mesh->nVertices(); i++)
            {
                GCVertex v_i = mesh->vertex(i);
                Vector3 sumDerivs{0, 0, 0};

                // Each vertex produces a derivative wrt its surrounding faces
                for (GCFace f : v_i.adjacentFaces())
                {
                    if (f.isBoundaryLoop()) continue;
                    sumDerivs += SurfaceDerivs::triangleAreaWrtVertex(geom, f, v_i);
                }
                // Differential of A^2 = 2 A (dA/dx)
                sumDerivs = 2 * currentValue * sumDerivs;
                MatrixUtils::addToRow(output, inds[v_i], weight * sumDerivs);
            }
        }
    }

    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void SoftAreaConstraint::Update()
    {
        // Nothing needs to be done
    }

    // Get the mesh associated with this energy.
    MeshPtr SoftAreaConstraint::GetMesh()
    {
        return mesh;
    }

    // Get the geometry associated with this geometry.
    GeomPtr SoftAreaConstraint::GetGeom()
    {
        return geom;
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 SoftAreaConstraint::GetExponents()
    {
        return Vector2{1, 0};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    BVHNode6D* SoftAreaConstraint::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double SoftAreaConstraint::GetTheta() {
        return 0;
    }

} // namespace rsurfaces