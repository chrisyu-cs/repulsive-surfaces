#include "energy/squared_error.h"
#include "matrix_utils.h"

namespace rsurfaces
{
    SquaredError::SquaredError(MeshPtr mesh_, GeomPtr geom_, double weight_)
    {
        mesh = mesh_;
        geom = geom_;
        weight = weight_;

        // Record initial positions
        originalPositions.ResetData(mesh);
        for (GCVertex v : mesh->vertices())
        {
            originalPositions[v] = geom->inputVertexPositions[v];
        }
    }

    // Returns the current value of the energy.
    double SquaredError::Value()
    {
        double sum = 0;
        // Sum of all squared deviations from original positions
        #pragma omp parallel for reduction(+ : sum)
        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            GCVertex v = mesh->vertex(i);
            double dist2 = norm2(originalPositions[v] - geom->inputVertexPositions[v]);
            sum += dist2;
        }
        return weight * sum;
    }

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void SquaredError::Differential(Eigen::MatrixXd &output)
    {
        VertexIndices inds = mesh->getVertexIndices();
        
        // Gradient of wx^2 = 2wx
        #pragma omp parallel shared(output)
        {
            #pragma omp parallel for
            for (size_t i = 0; i < mesh->nVertices(); i++)
            {
                GCVertex v = mesh->vertex(i);
                Vector3 disp = geom->inputVertexPositions[v] - originalPositions[v];
                MatrixUtils::addToRow(output, inds[v], 2 * weight * disp);
            }
        }
    }

    void SquaredError::ChangeVertexTarget(GCVertex v, Vector3 newPos)
    {
        originalPositions[v] = newPos;
    }
    
    // Get the mesh associated with this energy.
    MeshPtr SquaredError::GetMesh()
    {
        return mesh;
    }

    // Get the geometry associated with this geometry.
    GeomPtr SquaredError::GetGeom()
    {
        return geom;
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 SquaredError::GetExponents()
    {
        return Vector2{2, 0};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    OptimizedClusterTree* SquaredError::GetBVH()
    {
        return 0;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double SquaredError::GetTheta() {
        return 0;
    }

} // namespace rsurfaces