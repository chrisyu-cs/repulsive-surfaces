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

    struct MassNormalPoint
    {
        double mass;
        Vector3 normal;
        Vector3 point;
        Vector3 minCoords;
        Vector3 maxCoords;
        size_t elementID;
    };

    class SurfaceEnergy
    {
    public:
        virtual ~SurfaceEnergy() {}
        // Returns the current value of the energy.
        virtual double Value() = 0;
        // Returns the current differential of the energy, stored in the given
        // V x 3 matrix, where each row holds the differential (a 3-vector) with
        // respect to the corresponding vertex.
        virtual void Differential(Eigen::MatrixXd &output) = 0;
        // Update the energy to reflect the current state of the mesh. This could
        // involve building a new BVH for Barnes-Hut energies, for instance.
        virtual void Update() = 0;
        // Get the mesh associated with this energy.
        virtual MeshPtr GetMesh() = 0;
        // Get the geometry associated with this geometry.
        virtual GeomPtr GetGeom() = 0;
        // Get the exponents of this energy; only applies to tangent-point energies.
        virtual Vector2 GetExponents() = 0;
        // Get a pointer to the current BVH for this energy.
        // Return 0 if the energy doesn't use a BVH.
        virtual BVHNode6D *GetBVH() = 0;
    };

} // namespace rsurfaces