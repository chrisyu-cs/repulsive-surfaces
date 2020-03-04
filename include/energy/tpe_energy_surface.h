#pragma once

#include "spatial/mass_point.h"

namespace rsurfaces
{
using namespace geometrycentral;

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
    Vector3 tpe_gradient_Kf(GCFace f1, MassPoint p2, GCVertex wrt);
    Vector3 tpe_gradient_pair_num(GCFace f1, GCFace f2, GCVertex wrt, double eps);

};

} // namespace rsurfaces
