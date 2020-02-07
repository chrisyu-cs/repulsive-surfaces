#include "geometrycentral/utilities/vector3.h"
#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

namespace rsurfaces {
    using namespace geometrycentral;
    typedef std::unique_ptr<surface::HalfedgeMesh> MeshPtr;
    typedef std::unique_ptr<surface::VertexPositionGeometry> GeomPtr;
    typedef surface::Vertex GCVertex;
    typedef surface::Halfedge GCHalfedge;
    typedef surface::Edge GCEdge;
    typedef surface::Face GCFace;
}