#pragma once

#include "energy/tpe_kernel.h"
#include "spatial/bvh_3d.h"

namespace rsurfaces
{

class BarnesHutTPEnergy : public SurfaceEnergy
{
public:
    BarnesHutTPEnergy(TPEKernel *kernel_, BVHNode3D *root_);
    virtual double Value();
    virtual void Differential(Eigen::MatrixXd &output);
    virtual MeshPtr GetMesh();
    virtual GeomPtr GetGeom();

private:
    TPEKernel *kernel;
    BVHNode3D *root;
    double computeEnergyOfFace(GCFace face, BVHNode3D *bvhRoot);
    void addV(BVHNode3D *bvhRoot, Eigen::MatrixXd &V, VertexIndices &indices);
    void addVOfFace(GCFace face, BVHNode3D *node, Eigen::MatrixXd &V, VertexIndices &indices);
    void addW(BVHNode3D *bvhRoot, Eigen::MatrixXd &W, VertexIndices &indices);
    void addWForAllClusters(BVHNode3D *node, Eigen::MatrixXd &W, std::vector<double> &xi,
                            std::vector<Vector3> &eta, VertexIndices &indices);
    void accumulateWValues(GCFace face, BVHNode3D *node, std::vector<double> &xi,
                           std::vector<Vector3> &eta);
};

} // namespace rsurfaces