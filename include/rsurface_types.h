#pragma once

#include "geometrycentral/utilities/vector3.h"
#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <Eigen/Core>

namespace rsurfaces
{
using namespace geometrycentral;
typedef std::unique_ptr<surface::HalfedgeMesh> MeshUPtr;
typedef std::unique_ptr<surface::VertexPositionGeometry> GeomUPtr;
typedef std::shared_ptr<surface::HalfedgeMesh> MeshPtr;
typedef std::shared_ptr<surface::VertexPositionGeometry> GeomPtr;
typedef surface::Vertex GCVertex;
typedef surface::Halfedge GCHalfedge;
typedef surface::Edge GCEdge;
typedef surface::Face GCFace;


class SurfaceEnergy {
    public:
    virtual ~SurfaceEnergy() {}
    // Returns the current value of the energy.
    virtual double Value() = 0;
    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    virtual void Differential(Eigen::MatrixXd &output) = 0;
};

} // namespace rsurfaces