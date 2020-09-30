#pragma once

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/direction_fields.h"

#include <queue>

#include "rsurface_types.h"

namespace rsurfaces
{
    namespace remeshing
    {
        using namespace geometrycentral;
        using namespace geometrycentral::surface;
        
        bool shouldFlip(Edge e);
        
        void adjustVertexDegrees(MeshPtr const &mesh);

        bool isDelaunay(GeomPtr const &geometry, Edge e);

        void fixDelaunay(MeshPtr const &mesh, GeomPtr const &geometry);

        void smoothByLaplacian(MeshPtr const &mesh, GeomPtr const &geometry);

        Vector3 findcircumcenter(Vector3 p1, Vector3 p2, Vector3 p3);

        Vector3 findCircumcenter(GeomPtr const &geometry, Face f);

        void smoothByCircumcenter(MeshPtr const &mesh, GeomPtr const &geometry);
    } // namespace remeshing
} // namespace rsurfaces
