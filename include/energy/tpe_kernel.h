#pragma once

#include "spatial/bvh_types.h"

namespace rsurfaces
{
using namespace geometrycentral;

class TPEKernel
{
public:
    TPEKernel(const MeshPtr &m, const GeomPtr &g, double alpha, double beta);
    double tpe_pair(GCFace f1, GCFace f2);
    double tpe_pair(GCFace f1, MassPoint p2);
    Vector3 tpe_gradient_pair(GCFace f1, GCFace f2, GCVertex wrt);
    void numericalTest();
    Vector3 tpe_Kf_partial_wrt_v1(Vector3 v1, Vector3 v2, Vector3 n1);
    Vector3 tpe_Kf_partial_wrt_v2(Vector3 v1, Vector3 v2, Vector3 n1);
    Vector3 tpe_Kf_partial_wrt_n1(Vector3 v1, Vector3 v2, Vector3 n1);

    double tpe_Kf(Vector3 v1, Vector3 v2, Vector3 n1);
    double tpe_Kf(GCFace f1, GCFace f2);

    MeshPtr mesh;
    GeomPtr geom;
    double alpha, beta;

    private:
    Vector3 tpe_gradient_Kf(GCFace f1, GCFace f2, GCVertex wrt);
    Vector3 tpe_gradient_pair_num(GCFace f1, GCFace f2, GCVertex wrt, double eps);

};

} // namespace rsurfaces
