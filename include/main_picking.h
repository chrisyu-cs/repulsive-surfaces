#include "rsurface_types.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "polyscope/pick.h"

namespace rsurfaces
{
    /*
    * Element indices are enumerated densely in the order of
    * {vertex,face,edge,halfedge/corner}
    * [citation: personal communications with Nick Sharp]
    */

    inline bool tryGetPickedVertex(polyscope::SurfaceMesh *s, size_t i, MeshPtr mesh, GCVertex &vert)
    {
        if (!s)
            return false;
        if (i < s->nVertices())
        {
            vert = mesh->vertex(i);
            return true;
        }
        else
            return false;
    }

    inline bool tryGetPickedFace(polyscope::SurfaceMesh *s, size_t i, MeshPtr mesh, GCFace &face)
    {
        if (!s)
            return false;

        // Offset down by number of vertices
        i -= mesh->nVertices();

        if (0 <= i && i < mesh->nFaces())
        {
            face = mesh->face(i);
            return true;
        }
        else
            return false;
    }

    inline bool tryGetPickedEdge(polyscope::SurfaceMesh *s, size_t i, MeshPtr mesh, GCEdge &edge)
    {
        if (!s)
            return false;

        // Offset down by number of vertices
        i -= mesh->nVertices();
        i -= mesh->nFaces();

        if (0 <= i && i < mesh->nEdges())
        {
            edge = mesh->edge(i);
            return true;
        }
        else
            return false;
    }

    inline bool tryGetPickedHalfedge(polyscope::SurfaceMesh *s, size_t i, MeshPtr mesh, GCHalfedge &halfedge)
    {
        if (!s)
            return false;

        // Offset down by number of vertices
        i -= mesh->nVertices();
        i -= mesh->nFaces();
        i -= mesh->nEdges();

        if (0 <= i && i < mesh->nHalfedges())
        {
            halfedge = mesh->halfedge(s->halfedgePerm[i]);
            return true;
        }
        else
            return false;
    }

    inline Vector2 projectToScreenCoords(Vector3 pos, glm::mat4 viewProj)
    {
        glm::vec4 gv{pos.x, pos.y, pos.z, 1};
        gv = viewProj * gv;

        auto io = ImGui::GetIO();
        glm::vec4 perspDiv = gv / gv.w;

        Vector2 unscaled{io.DisplaySize.x * (1 + perspDiv.x) / 2, io.DisplaySize.y * ((1 - perspDiv.y) / 2)};
        return Vector2{io.DisplayFramebufferScale.x * unscaled.x, io.DisplayFramebufferScale.y * unscaled.y};
    }

    inline Vector3 projectToScreenCoords3(Vector3 pos, glm::mat4 viewProj)
    {
        glm::vec4 gv{pos.x, pos.y, pos.z, 1};
        gv = viewProj * gv;

        auto io = ImGui::GetIO();
        glm::vec4 perspDiv = gv / gv.w;

        Vector2 unscaled{io.DisplaySize.x * (1 + perspDiv.x) / 2, io.DisplaySize.y * ((1 - perspDiv.y) / 2)};
        return Vector3{io.DisplayFramebufferScale.x * unscaled.x, io.DisplayFramebufferScale.y * unscaled.y, perspDiv.z};
    }

    inline Vector2 getMouseScreenPos()
    {
        auto io = ImGui::GetIO();
        ImVec2 p = ImGui::GetMousePos();
        Vector2 screenPos{io.DisplayFramebufferScale.x * p.x, io.DisplayFramebufferScale.y * p.y};
        return screenPos;
    }

    inline Vector3 unprojectFromScreenCoords3(Vector2 pos, double depth, glm::mat4 viewProj)
    {
        auto io = ImGui::GetIO();
        double normalized_x = 2 * (pos.x / (io.DisplaySize.x * io.DisplayFramebufferScale.x)) - 1;
        double normalized_y = 1 - 2 * (pos.y / (io.DisplaySize.y * io.DisplayFramebufferScale.y));

        glm::vec4 g{normalized_x, normalized_y, depth, 1};
        g = glm::inverse(viewProj) * g;
        g /= g.w;

        return Vector3{g.x, g.y, g.z};
    }

    inline GCVertex nearestVertexToScreenPos(Vector2 screenPos, GeomPtr geom, glm::mat4 viewProj, GCVertex v)
    {
        Vector2 screen = projectToScreenCoords(geom->inputVertexPositions[v], viewProj);
        return v;
    }

    inline GCVertex nearestVertexToScreenPos(Vector2 screenPos, GeomPtr geom, glm::mat4 viewProj, GCHalfedge e)
    {
        GCVertex v1 = e.vertex();
        GCVertex v2 = e.twin().vertex();
        Vector2 screen_v1 = projectToScreenCoords(geom->inputVertexPositions[v1], viewProj);
        Vector2 screen_v2 = projectToScreenCoords(geom->inputVertexPositions[v2], viewProj);

        if (norm2(screen_v1 - screenPos) < norm2(screen_v2 - screenPos))
        {
            return v1;
        }
        else
            return v2;
    }

    inline GCVertex nearestVertexToScreenPos(Vector2 screenPos, GeomPtr geom, glm::mat4 viewProj, GCEdge e)
    {
        return nearestVertexToScreenPos(screenPos, geom, viewProj, e.halfedge());
    }

    inline GCVertex nearestVertexToScreenPos(Vector2 screenPos, GeomPtr geom, glm::mat4 viewProj, GCFace f)
    {
        GCVertex nearest = f.halfedge().vertex();
        GCVertex first = nearest;
        double nearestDist = norm2(projectToScreenCoords(geom->inputVertexPositions[nearest], viewProj) - screenPos);

        for (GCVertex v : f.adjacentVertices())
        {
            if (v == first)
                continue;
            double v_dist = norm2(projectToScreenCoords(geom->inputVertexPositions[v], viewProj) - screenPos);
            if (v_dist < nearestDist)
            {
                nearest = v;
                nearestDist = v_dist;
            }
        }

        return nearest;
    }

} // namespace rsurfaces
