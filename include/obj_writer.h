#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
    void writeMeshToOBJ(MeshPtr mesh, GeomPtr geom, std::string output);
}