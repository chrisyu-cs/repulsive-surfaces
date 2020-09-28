#include "energy/static_obstacle.h"
#include "matrix_utils.h"

#include <omp.h>

namespace rsurfaces
{
    StaticObstacle::StaticObstacle(MeshPtr mesh_, GeomPtr geom_, MeshUPtr obsMesh_, GeomUPtr obsGeom_, double p_, double theta_, double weight_)
        : obstacleMesh(std::move(obsMesh_)), obstacleGeom(std::move(obsGeom_))
    {
        mesh = mesh_;
        geom = geom_;
        p = p_;
        theta = theta_;
        weight = weight_;

        obstacleBvh = 0;

        // We don't expect the obstacle to ever change shape,
        // so just initialize the BVH here
        Update();
    }

    inline double repulsivePotential(Vector3 x, Vector3 y, double p)
    {
        return 1.0 / pow(norm(x - y), p);
    }

    inline Vector3 repulsiveForce(Vector3 x, Vector3 y, double p)
    {
        Vector3 dir = y - x;
        double dist = dir.norm();
        dir /= dist;
        return dir * p / pow(dist, p + 1);
    }

    // Returns the current value of the energy.
    double StaticObstacle::Value()
    {
        double sum = 0;

        #pragma omp parallel for reduction(+ : sum) shared(obstacleBvh)
        for (size_t i = 0; i < mesh->nVertices(); i++)
        {
            GCVertex v = mesh->vertex(i);
            sum += computeEnergyOfVertex(v, obstacleBvh);
        }
        return sum;
    }

    double StaticObstacle::computeEnergyOfVertex(GCVertex vertex, BVHNode6D *bvhRoot)
    {
        Vector3 x = geom->inputVertexPositions[vertex];
        double mass = geom->vertexDualAreas[vertex];

        if (bvhRoot->nodeType == BVHNodeType::Empty)
        {
            return 0;
        }
        else if (bvhRoot->nodeType == BVHNodeType::Leaf)
        {
            // Compute the energy exactly for the one face in the cluster
            Vector3 node_x = bvhRoot->centerOfMass;
            double nodeMass = bvhRoot->totalMass;
            return nodeMass * repulsivePotential(x, node_x, p);
        }
        else if (bvhRoot->isAdmissibleFrom(x, theta))
        {
            // Use the cluster approximation
            MassNormalPoint mnp = bvhRoot->GetMassNormalPoint();
            double nodeMass = bvhRoot->totalMass;
            return nodeMass * repulsivePotential(x, mnp.point, p);
        }
        else
        {
            // Recursively compute it on all children
            double sum = 0;
            for (BVHNode6D *child : bvhRoot->children)
            {
                sum += computeEnergyOfVertex(vertex, child);
            }
            return sum;
        }
    }

    // Returns the current differential of the energy, stored in the given
    // V x 3 matrix, where each row holds the differential (a 3-vector) with
    // respect to the corresponding vertex.
    void StaticObstacle::Differential(Eigen::MatrixXd &output)
    {
        VertexIndices inds = mesh->getVertexIndices();
        Eigen::MatrixXd partialOutput;
        partialOutput.setZero(output.rows(), 3);

        #pragma omp parallel shared(obstacleBvh, output) firstprivate(partialOutput)
        {
            #pragma omp parallel for
            for (size_t i = 0; i < mesh->nVertices(); i++)
            {
                GCVertex v = mesh->vertex(i);
                Vector3 force = computeForceAtVertex(v, obstacleBvh);
                MatrixUtils::addToRow(partialOutput, inds[v], force);
            }

            #pragma omp critical
            {
                output += partialOutput;
            }
        }
    }

    Vector3 StaticObstacle::computeForceAtVertex(GCVertex vertex, BVHNode6D *bvhRoot)
    {
        Vector3 x = geom->inputVertexPositions[vertex];
        double mass = geom->vertexDualAreas[vertex];

        if (bvhRoot->nodeType == BVHNodeType::Empty)
        {
            return Vector3{0, 0, 0};
        }
        else if (bvhRoot->nodeType == BVHNodeType::Leaf)
        {
            // Compute the force exactly for the one face in the cluster
            Vector3 node_x = bvhRoot->centerOfMass;
            double nodeMass = bvhRoot->totalMass;
            return nodeMass * repulsiveForce(x, node_x, p);
        }
        else if (bvhRoot->isAdmissibleFrom(x, theta))
        {
            // Use the cluster approximation
            MassNormalPoint mnp = bvhRoot->GetMassNormalPoint();
            double nodeMass = bvhRoot->totalMass;
            return nodeMass * repulsiveForce(x, mnp.point, p);
        }
        else
        {
            // Recursively compute it on all children
            Vector3 sum{0, 0, 0};
            for (BVHNode6D *child : bvhRoot->children)
            {
                sum += computeForceAtVertex(vertex, child);
            }
            return sum;
        }
    }

    // Update the energy to reflect the current state of the mesh. This could
    // involve building a new BVH for Barnes-Hut energies, for instance.
    void StaticObstacle::Update()
    {
        if (obstacleBvh)
        {
            delete obstacleBvh;
        }
        obstacleBvh = Create6DBVHFromMeshVerts(obstacleMesh, obstacleGeom);
    }

    // Get the mesh associated with this energy.
    MeshPtr StaticObstacle::GetMesh()
    {
        return mesh;
    }

    // Get the geometry associated with this geometry.
    GeomPtr StaticObstacle::GetGeom()
    {
        return geom;
    }

    // Get the exponents of this energy; only applies to tangent-point energies.
    Vector2 StaticObstacle::GetExponents()
    {
        return Vector2{0, p};
    }

    // Get a pointer to the current BVH for this energy.
    // Return 0 if the energy doesn't use a BVH.
    BVHNode6D *StaticObstacle::GetBVH()
    {
        return obstacleBvh;
    }

    // Return the separation parameter for this energy.
    // Return 0 if this energy doesn't do hierarchical approximation.
    double StaticObstacle::GetTheta()
    {
        return theta;
    }

} // namespace rsurfaces