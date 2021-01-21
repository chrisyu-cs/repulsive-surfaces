#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
    void writeMeshToOBJ(MeshPtr mesh, GeomPtr geom, GeomPtr geomOrig, bool writeAreaRatios, std::string output);
}
