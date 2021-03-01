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

    typedef surface::VertexData<size_t> VertexIndices;
    typedef surface::FaceData<size_t> FaceIndices;

    class BVHNode6D;

    struct MassPoint
    {
        double mass;
        Vector3 point;
        size_t elementID;
    };

    enum class GradientMethod
    {
        HsProjected,
        HsProjectedIterative,
        HsExactProjected,
        H1Projected,
        L2Unconstrained,
        L2Projected,
        AQP,
        H1_LBFGS,
        BQN_LBFGS
    };

    struct MassNormalPoint
    {
        double mass;
        Vector3 normal;
        Vector3 point;
        Vector3 minCoords;
        Vector3 maxCoords;
        size_t elementID;
    };

} // namespace rsurfaces