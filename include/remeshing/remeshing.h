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
        
        void collapseEdge(MeshPtr const &mesh, GeomPtr const &geometry, Edge e);
        
        void testStuff(MeshPtr const &mesh, GeomPtr const &geometry, int edgeIndex);
        
        bool shouldFlip(Edge e);
        
        void adjustVertexDegrees(MeshPtr const &mesh);

        bool isDelaunay(GeomPtr const &geometry, Edge e);

        void fixDelaunay(MeshPtr const &mesh, GeomPtr const &geometry);

        void smoothByLaplacian(MeshPtr const &mesh, GeomPtr const &geometry);

        Vector3 findCircumcenter(Vector3 p1, Vector3 p2, Vector3 p3);

        Vector3 findCircumcenter(GeomPtr const &geometry, Face f);

        void smoothByCircumcenter(MeshPtr const &mesh, GeomPtr const &geometry);
        
        void adjustEdgeLengths(MeshPtr const &mesh, GeomPtr const &geometry);
        
        Vector3 findBarycenter(Vector3 p1, Vector3 p2, Vector3 p3);

        Vector3 findBarycenter(GeomPtr const &geometry, Face f);
        
        void smoothByFaceWeight(MeshPtr const &mesh, GeomPtr const &geometry, FaceData<double> faceWeight);
    } // namespace remeshing
} // namespace rsurfaces
