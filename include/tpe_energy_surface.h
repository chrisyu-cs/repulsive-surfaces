#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
using namespace geometrycentral;

inline Vector3 faceBarycenter(GeomPtr const &geom, GCFace f)
{
    Vector3 sum{0, 0, 0};
    int count = 0;
    for (GCVertex v : f.adjacentVertices())
    {
        sum += geom->inputVertexPositions[v];
        count++;
    }
    return sum / count;
}

class TPEKernel
{
public:
    TPEKernel(const MeshPtr &m, const GeomPtr &g, double alpha, double beta);
    double tpe_pair(GCFace f1, GCFace f2);
    Vector3 tpe_gradient_pair(GCFace f1, GCFace f2, GCVertex wrt);

    void numericalTest();

    MeshPtr mesh;
    GeomPtr geom;
    double alpha, beta;

    private:
    double tpe_Kf(GCFace f1, GCFace f2);
    Vector3 tpe_gradient_Kf(GCFace f1, GCFace f2, GCVertex wrt);
    Vector3 tpe_gradient_Kf_num(GCFace f1, GCFace f2, GCVertex wrt, double eps);

};

} // namespace rsurfaces
