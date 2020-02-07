#include "tpe_energy_surface.h"

namespace rsurfaces {

    SurfaceTPE::SurfaceTPE(MeshPtr m, GeomPtr g, double a, double b)
    : mesh(std::move(m)), geom(std::move(g))  {
        alpha = a;
        beta = b;
    }

    inline Vector3 faceBarycenter(GeomPtr const &geom, GCFace f) {
        Vector3 sum{0, 0, 0};
        int count = 0;
        for (GCVertex v : f.adjacentVertices()) {
            sum += geom->vertexPositions[v];
            count++;
        }
        return sum / count;
    }

    double SurfaceTPE::tpe_pair(GCFace f1, GCFace f2) {
        Vector3 n1 = geom->faceNormal(f1);
        Vector3 v1 = faceBarycenter(geom, f1);
        Vector3 v2 = faceBarycenter(geom, f2);
        double w1 = geom->faceArea(f1);
        double w2 = geom->faceArea(f2);

        Vector3 displacement = v2 - v1;
        double numer = pow(dot(displacement, n1), alpha);
        double denom = pow(displacement.norm(), beta);

        return (numer / denom) * w1 * w2;
    }
}
