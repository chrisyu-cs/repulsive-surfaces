#include "fractional_laplacian.h"
#include "helpers.h"
#include "surface_derivatives.h"
#include "energy/tpe_energy_surface.h"

namespace rsurfaces
{
Vector3 FractionalLaplacian::HatGradientOnFace(const GeomPtr &geom, GCFace face, GCVertex vertex)
{
    GCHalfedge he;
    bool found = findVertexInTriangle(face, vertex, he);
    if (!found)
    {
        return Vector3{0, 0, 0};
    }

    GCHalfedge opp = he.next();
    GCVertex next = opp.vertex();
    Vector3 oppDir = geom->inputVertexPositions[opp.twin().vertex()] - geom->inputVertexPositions[opp.vertex()];
    oppDir = oppDir.normalize();

    Vector3 perp = geom->inputVertexPositions[vertex] - geom->inputVertexPositions[next];
    perp = perp - dot(oppDir, perp) * oppDir;
    // We want (p / |p|) * (1 / |p|): in the orthogonal direction,
    // with magnitude inversely proportional to the orthogonal distance
    perp = perp / perp.norm2();

    return perp;
}

void FractionalLaplacian::AddTriPairValues(const GeomPtr &geom, GCFace S, GCFace T, double s_pow,
                                          surface::VertexData<size_t> &indices, Eigen::MatrixXd &A)
{
    std::vector<GCVertex> verts;
    GetVerticesWithoutDuplicates(S, T, verts);
    double area_S = geom->faceArea(S);
    double area_T = geom->faceArea(T);

    for (GCVertex u : verts)
    {
        for (GCVertex v : verts)
        {
            Vector3 u_hat_s = HatGradientOnFace(geom, S, u);
            Vector3 u_hat_t = HatGradientOnFace(geom, T, u);
            Vector3 v_hat_s = HatGradientOnFace(geom, S, v);
            Vector3 v_hat_t = HatGradientOnFace(geom, T, v);

            double dot_term = dot(u_hat_s - u_hat_t, v_hat_s - v_hat_t);
            Vector3 f_S = faceBarycenter(geom, S);
            Vector3 f_T = faceBarycenter(geom, T);
            double dist_term = 1.0 / pow(norm(f_S - f_T), 2 * (s_pow - 1) + 3);
            
            A(indices[u], indices[v]) += dot_term * dist_term * area_S * area_T;
        }
    }
}

void FractionalLaplacian::FillMatrix(const MeshPtr &mesh, const GeomPtr &geom, double s_pow, Eigen::MatrixXd &A) {
    surface::VertexData<size_t> indices = mesh->getVertexIndices();
    
    for (GCFace S : mesh->faces()) {
        for (GCFace T : mesh->faces()) {
            if (S == T) continue;
            AddTriPairValues(geom, S, T, s_pow, indices, A);
        }
    }
}

} // namespace rsurfaces