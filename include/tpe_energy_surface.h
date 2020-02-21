#include "rsurface_types.h"

namespace rsurfaces
{
using namespace geometrycentral;

class SurfaceTPE
{
public:
    SurfaceTPE(MeshPtr m, GeomPtr g, double alpha, double beta);
    double tpe_pair(GCFace v1, GCFace v2);

private:
    MeshPtr mesh;
    GeomPtr geom;
    double alpha, beta;
};

} // namespace rsurfaces
