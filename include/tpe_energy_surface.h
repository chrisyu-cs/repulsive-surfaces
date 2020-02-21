#include "rsurface_types.h"

namespace rsurfaces
{
using namespace geometrycentral;

class SurfaceTPE
{
public:
    SurfaceTPE(MeshPtr m, GeomPtr g, double alpha, double beta);
    double tpe_pair(GCFace f1, GCFace f2);
    double tpe_gradient(GCFace f1, GCFace f2, GCVertex wrt);

private:
    MeshPtr mesh;
    GeomPtr geom;
    double alpha, beta;
};

} // namespace rsurfaces
