#include "remeshing/remeshing.h"

namespace rsurfaces
{
    namespace remeshing
    {
        using std::cout;
        using std::queue;

        inline Vector3 projectToPlane(Vector3 v, Vector3 norm)
        {
            return v - norm * dot(norm, v);
        }

        bool shouldFlip(GeomPtr const &geometry, Edge e) {
            // Check if, by flipping this edge, you would make the 
            // degrees of all the vertices on this diamond closer to 6

            // Vertices v1, v2, v3, v4 with degrees d1, d2, d3, d4
            // f(d1, d2, d3, d4) = |d1 - 6| + |d2 - 6| + |d3 - 6| + |d4 - 6|

            // Do the flip if it would reduce the value of f on this diamond after flipping
        }

        bool isDelaunay(GeomPtr const &geometry, Edge e)
        {
            float angle1 = geometry->cornerAngle(e.halfedge().next().next().corner());
            float angle2 = geometry->cornerAngle(e.halfedge().twin().next().next().corner());
            return angle1 + angle2 <= PI;
        }

        void fixDelaunay(MeshPtr const &mesh, GeomPtr const &geometry)
        {
            // queue of edges to check if Delaunay
            queue<Edge> toCheck;
            // true if edge is currently in toCheck
            EdgeData<bool> inQueue(*mesh);
            // start with ALL edges
            for (Edge e : mesh->edges())
            {
                toCheck.push(e);
                inQueue[e] = true;
            }
            // counter and limit for number of flips
            int flipMax = 100 * mesh->nVertices();
            int flipCnt = 0;
            while (!toCheck.empty() && flipCnt < flipMax)
            {
                Edge e = toCheck.front();
                toCheck.pop();
                inQueue[e] = false;
                // if not Delaunay, flip edge and enqueue the surrounding "diamond" edges (if not already)
                if (!isDelaunay(geometry, e))
                {
                    flipCnt++;
                    Halfedge he = e.halfedge();
                    Halfedge he1 = he.next();
                    Halfedge he2 = he1.next();
                    Halfedge he3 = he.twin().next();
                    Halfedge he4 = he3.next();

                    if (!inQueue[he1.edge()])
                    {
                        toCheck.push(he1.edge());
                        inQueue[he1.edge()] = true;
                    }
                    if (!inQueue[he2.edge()])
                    {
                        toCheck.push(he2.edge());
                        inQueue[he2.edge()] = true;
                    }
                    if (!inQueue[he3.edge()])
                    {
                        toCheck.push(he3.edge());
                        inQueue[he3.edge()] = true;
                    }
                    if (!inQueue[he4.edge()])
                    {
                        toCheck.push(he4.edge());
                        inQueue[he4.edge()] = true;
                    }
                    mesh->flip(e);
                }
            }
        }

        void smoothByLaplacian(MeshPtr const &mesh, GeomPtr const &geometry)
        {
            // smoothed vertex positions
            VertexData<Vector3> newVertexPosition(*mesh);
            for (Vertex v : mesh->vertices())
            {
                // calculate average of surrounding vertices
                newVertexPosition[v] = Vector3::zero();
                for (Vertex j : v.adjacentVertices())
                {
                    newVertexPosition[v] += geometry->inputVertexPositions[j];
                }
                newVertexPosition[v] /= v.degree();
                // and project the average to the tangent plane
                Vector3 updateDirection = newVertexPosition[v] - geometry->inputVertexPositions[v];
                updateDirection = projectToPlane(updateDirection, geometry->vertexNormals[v]);
                newVertexPosition[v] = geometry->inputVertexPositions[v] + updateDirection;
            }
            // update final vertices
            for (Vertex v : mesh->vertices())
            {
                geometry->inputVertexPositions[v] = newVertexPosition[v];
            }
        }

        Vector3 findCircumcenter(Vector3 p1, Vector3 p2, Vector3 p3)
        {
            // barycentric coordinates of circumcenter
            double a = (p3 - p2).norm();
            double b = (p3 - p1).norm();
            double c = (p2 - p1).norm();
            double a2 = a * a;
            double b2 = b * b;
            double c2 = c * c;
            Vector3 O{a2 * (b2 + c2 - a2), b2 * (c2 + a2 - b2), c2 * (a2 + b2 - c2)};
            // normalize to sum of 1
            O /= O[0] + O[1] + O[2];
            // change back to space
            return O[0] * p1 + O[1] * p2 + O[2] * p3;
        }

        Vector3 findCircumcenter(GeomPtr const &geometry, Face f)
        {
            // retrieve the face's vertices
            int index = 0;
            Vector3 p[3];
            for (Vertex v0 : f.adjacentVertices())
            {
                p[index] = geometry->inputVertexPositions[v0];
                index++;
            }
            return findCircumcenter(p[0], p[1], p[2]);
        }

        void smoothByCircumcenter(MeshPtr const &mesh, GeomPtr const &geometry)
        {
            // smoothed vertex positions
            VertexData<Vector3> newVertexPosition(*mesh);
            for (Vertex v : mesh->vertices())
            {
                newVertexPosition[v] = Vector3::zero();
                Vector3 updateDirection = Vector3::zero();
                // for each face
                for (Face f : v.adjacentFaces())
                {
                    // add the circumcenter weighted by face area to the update direction
                    Vector3 circum = findCircumcenter(geometry, f);
                    updateDirection += geometry->faceAreas[f] * (circum - geometry->inputVertexPositions[v]);
                }
                // project update direction to tangent plane
                updateDirection = projectToPlane(updateDirection, geometry->vertexNormals[v]);
                newVertexPosition[v] = geometry->inputVertexPositions[v] + updateDirection;
            }
            // update final vertices
            for (Vertex v : mesh->vertices())
            {
                geometry->inputVertexPositions[v] = newVertexPosition[v];
            }
        }
    } // namespace remeshing
} // namespace rsurfaces