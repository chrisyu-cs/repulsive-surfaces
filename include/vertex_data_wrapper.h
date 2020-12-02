#pragma once

#include "rsurface_types.h"

namespace rsurfaces
{
    class VertexDataWrapper
    {
    public:
        geometrycentral::surface::VertexData<Vector3> data;

        Vector3 const &operator[](GCVertex i) const
        {
            return data[i];
        }

        Vector3 &operator[](GCVertex i)
        {
            return data[i];
        }

        void ResetData(MeshPtr mesh)
        {
            data.clear();
            data = geometrycentral::surface::VertexData<Vector3>(*mesh);
        }
    };
} // namespace rsurfaces