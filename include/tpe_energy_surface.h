#include "rsurface_types.h"

namespace rsurfaces
{
using namespace geometrycentral;

class SurfaceTPE
{
public:
    SurfaceTPE(MeshPtr m, GeomPtr g, double alpha, double beta);
    double tpe_Kf(GCFace f1, GCFace f2);
    double tpe_pair(GCFace f1, GCFace f2);
    Vector3 tpe_gradient_Kf(GCFace f1, GCFace f2, GCVertex wrt);
    Vector3 tpe_gradient_Kf_num(GCFace f1, GCFace f2, GCVertex wrt, double eps);
    Vector3 tpe_gradient_pair(GCFace f1, GCFace f2, GCVertex wrt);

    void numericalTest();

private:
    MeshPtr mesh;
    GeomPtr geom;
    double alpha, beta;
};

} // namespace rsurfaces
