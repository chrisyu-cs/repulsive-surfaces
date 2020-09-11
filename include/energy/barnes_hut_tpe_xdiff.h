#pragma once

#include "energy/tpe_kernel.h"
#include "spatial/bvh_6d.h"

namespace rsurfaces
{
    struct BHDiffData
    {
        Vector3 dCenter = Vector3{0, 0, 0};
        double dArea = 0;
    };

    // Evaluates energy and differential using Barnes-Hut with a BVH.
    class BarnesHutTPEnergyXDiff : public SurfaceEnergy
    {
    public:
        BarnesHutTPEnergyXDiff(TPEKernel *kernel_, double theta_);
        ~BarnesHutTPEnergyXDiff();
        virtual double Value();
        virtual void Differential(Eigen::MatrixXd &output);
        virtual void Update();
        virtual MeshPtr GetMesh();
        virtual GeomPtr GetGeom();
        virtual Vector2 GetExponents();
        virtual BVHNode6D *GetBVH();
        virtual double GetTheta();

    private:
        TPEKernel *kernel;
        BVHNode6D *root;
        double theta;
        double computeEnergyOfFace(GCFace face, BVHNode6D *bvhRoot);
        Vector3 diffKernelWrtSumCenters(Vector3 x_T, Vector3 n_T, Vector3 sumCenters, double sumAreas);
        Vector3 diffEnergyWrtSumCenters(GCFace face, Vector3 sumCenters, double sumAreas);
        double diffKernelWrtSumAreas(Vector3 x_T, Vector3 n_T, Vector3 sumCenters, double sumAreas);
        double diffEnergyWrtSumAreas(GCFace face, Vector3 sumCenters, double sumAreas);

        void percolateDiffsDown(DataTree<BHDiffData> *dataRoot, Eigen::MatrixXd &output,
                                surface::VertexData<size_t> &indices);
        void accumulateOneSidedGradient(Eigen::MatrixXd &gradients, BVHNode6D *node, GCFace face1,
                                        surface::VertexData<size_t> &indices);
        void accClusterGradients(BVHNode6D *bvhRoot, GCFace face, DataTreeContainer<BHDiffData> *data);
        void accFaceDerivative(BVHNode6D *bvhRoot, GCFace face, DataTreeContainer<BHDiffData> *data);
    };

} // namespace rsurfaces