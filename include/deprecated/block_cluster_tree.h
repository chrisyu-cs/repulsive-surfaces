#pragma once

#include "bct_kernel_type.h"
#include <omp.h>
#include "rsurface_types.h"
#include <Eigen/Core>
#include "spatial/bvh_6d.h"
#include "sobolev/hs_operators.h"
#include "helpers.h"

namespace rsurfaces
{

    struct ClusterPair
    {
        BVHNode6D *cluster1;
        BVHNode6D *cluster2;
    };

    struct PercolationData
    {
        double wtDot = 0;
        double B = 0;
    };

    struct AdmissibleClusterList
    {
        BVHNode6D *node = 0;
        std::vector<ClusterPair> list;
    };

    class BlockClusterTree
    {
    public:
        static long illSepTime;
        static long wellSepTime;
        static long traversalTime;

        BlockClusterTree(const MeshPtr &mesh, const GeomPtr &geom, BVHNode6D *root, double sepCoeff, double s_, double e = 0.0);
        ~BlockClusterTree();
        // Loop over all currently inadmissible cluster pairs
        // and subdivide them to their children.
        void splitInadmissibleNodes();
        static bool isPairAdmissible(ClusterPair pair, double coeff);
        static bool isPairSmallEnough(ClusterPair pair);

        inline size_t expectedNRows() const
        {
            return mesh->nVertices() * 3;
        }
        inline size_t expectedNCols() const
        {
            return mesh->nVertices() * 3;
        }

        void PrintData();
        void PrintAdmissibleClusters(std::ofstream &stream);
        void PrintInadmissibleClusters(std::ofstream &stream);

        template <typename V, typename Dest>
        void AfFullProduct(ClusterPair pair, const V &v_mid, Dest &result, BCTKernelType kType) const;

        template <typename V, typename Dest>
        void AfApproxProduct(ClusterPair pair, const V &v_mid, Dest &result, BCTKernelType kType) const;

        // Multiplies A * v, where v holds a vector3 at each vertex in a flattened column,
        //  and stores it in b.
        template <typename V3, typename Dest>
        void MultiplyVector3(const V3 &v, Dest &b, BCTKernelType kType, bool addToResult = false) const;

        // Same as the above but const because fuck eigen
        template <typename V3, typename Dest>
        void MultiplyVector3Const(const V3 &v, Dest &b, BCTKernelType kType, bool addToResult = false, double eps = 0) const;

        inline void SetExponent(double s_)
        {
            exp_s = s_;
        }

        inline void recomputeBarycenters() const
        {
            for (GCFace f : mesh->faces())
            {
                faceBarycenters[f] = faceBarycenter(geom, f);
            }
        }

        void PremultiplyAf1(BCTKernelType kType) const;

    private:
        // Multiplies A * v and stores it in b.
        template <typename V, typename Dest>
        void MultiplyVector(const V &v, Dest &b, BCTKernelType kType, bool addToResult = false) const;

        // Multiplies the inadmissible clusters for A * v, storing it in b.
        template <typename V, typename Dest>
        void MultiplyInadmissible(const V &v_hat, Dest &b_hat, BCTKernelType kType) const;

        // Multiplies the admissible clusters using a percolation method.
        template <typename V, typename Dest>
        void MultiplyAdmissiblePercolated(const V &v, Dest &b, BCTKernelType kType) const;

        // Multiplies just the kernel matrix A using a percolation method.
        template <typename V, typename Dest>
        void MultiplyAfPercolated(const V &v, Dest &b, BCTKernelType kType) const;

        inline double kernel(BCTKernelType kType, double exp_s, Vector3 v1, Vector3 v2, Vector3 n1, Vector3 n2) const
        {
            switch (kType)
            {
            case BCTKernelType::FractionalOnly:
                return Hs::MetricDistanceTermFrac(exp_s, v1, v2);
            case BCTKernelType::HighOrder:
                return Hs::MetricDistanceTerm(exp_s, v1, v2);
            case BCTKernelType::LowOrder:
                return Hs::MetricDistanceTermLowPure(exp_s, v1, v2, n1, n2);
            }
        }

        // Cached list of face barycenters to avoid recomputation
        mutable geometrycentral::surface::FaceData<Vector3> faceBarycenters;

        void fillClusterMasses(BVHNode6D *cluster, Eigen::VectorXd &w) const;
        void OrganizePairsByFirst();

        mutable Eigen::VectorXd Af_1_High;
        mutable bool highInitialized;
        mutable Eigen::VectorXd Af_1_Frac;
        mutable bool fracInitialized;
        mutable Eigen::VectorXd Af_1_Low;
        mutable bool lowInitialized;

        const Eigen::VectorXd &getPremultipliedAf1(BCTKernelType kType) const;

        double exp_s, separationCoeff;
        double epsilon;
        const MeshPtr mesh;
        const GeomPtr geom;
        BVHNode6D *tree_root;
        std::vector<ClusterPair> admissiblePairs;
        std::vector<std::vector<ClusterPair>> admissibleByCluster;

        std::vector<ClusterPair> unresolvedPairs;
        std::vector<ClusterPair> inadmissiblePairs;
    };

    template <typename V, typename Dest>
    void BlockClusterTree::MultiplyAdmissiblePercolated(const V &v, Dest &b, BCTKernelType kType) const
    {
        MultiplyAfPercolated(v, b, kType);
        b = 2 * (getPremultipliedAf1(kType).asDiagonal() * v - b);
    }

    template <typename V>
    // Get the dot product W^T * J for the given node in a data tree (and all children)
    void percolateWtDot(DataTree<PercolationData> *dataTree, const V &v, MeshPtr mesh, GeomPtr geom)
    {
        double rootSum = 0;
        // If this is a leaf, just set the value directly by multiplying weight * V
        if (dataTree->node->nodeType == BVHNodeType::Leaf)
        {
            rootSum = dataTree->node->totalMass * v(dataTree->node->elementID);
        }
        // Otherwise, go over all children and sum their values
        else
        {
            for (DataTree<PercolationData> *child : dataTree->children)
            {
                percolateWtDot(child, v, mesh, geom);
                rootSum += child->data.wtDot;
            }
        }
        dataTree->data.wtDot = rootSum;
    }

    template <typename Dest>
    void percolateJ(DataTree<PercolationData> *dataTree, double parentB, Dest &b)
    {
        dataTree->data.B += parentB;
        // If we've already percolated down to a leaf, then the leaf already
        // contains the value for the corresponding entry in b, so copy it in
        if (dataTree->node->nodeType == BVHNodeType::Leaf)
        {
            // The result ends up being diag(w) * b for the full vector,
            // but we can just multiply the diag(w) part here
            b(dataTree->node->elementID) = dataTree->node->totalMass * dataTree->data.B;
        }
        // Otherwise we need to percolate the node's B down to all children.
        // Assume that children already have sum of their admissible a_IJ * V_J in B
        for (DataTree<PercolationData> *child : dataTree->children)
        {
            // Percolate downward through the rest
            percolateJ(child, dataTree->data.B, b);
        }
    }

    template <typename V, typename Dest>
    void BlockClusterTree::MultiplyAfPercolated(const V &v, Dest &b, BCTKernelType kType) const
    {
        DataTreeContainer<PercolationData> *treeContainer = tree_root->CreateDataTree<PercolationData>();
        // Percolate W^T * v upward through the tree
        percolateWtDot(treeContainer->tree, v, mesh, geom);

// For each cluster I, we need to sum over all clusters J that are
// admissible with it. Since we already have a list of all admissible
// pairs, we can just do this for all clusters at once.
#pragma omp parallel for shared(admissibleByCluster, treeContainer)
        for (size_t i = 0; i < admissibleByCluster.size(); i++)
        {
            for (ClusterPair const &pair : admissibleByCluster[i])
            {
                double a_IJ = kernel(kType, exp_s, pair.cluster1->centerOfMass, pair.cluster2->centerOfMass,
                                     pair.cluster1->averageNormal, pair.cluster2->averageNormal);
                DataTree<PercolationData> *data_I = treeContainer->byIndex[pair.cluster1->nodeID];
                DataTree<PercolationData> *data_J = treeContainer->byIndex[pair.cluster2->nodeID];
                // Each I gets a sum of a_IJ * V_J for all admissible J
                data_I->data.B += a_IJ * data_J->data.wtDot;
            }
        }

        // Percolate downward from the root
        percolateJ(treeContainer->tree, 0, b);

        // Now the result is stored in b, so just clean up
        delete treeContainer;
    }

    template <typename V, typename Dest>
    void BlockClusterTree::MultiplyInadmissible(const V &v_hat, Dest &b_hat, BCTKernelType kType) const
    {
        Eigen::VectorXd result;
        result.setZero(b_hat.rows());

#pragma omp parallel firstprivate(result) shared(b_hat)
        {
#pragma omp for
            for (size_t i = 0; i < inadmissiblePairs.size(); i++)
            {
                AfFullProduct(inadmissiblePairs[i], v_hat, result, kType);
            }

#pragma omp critical
            {
                b_hat += result;
            }
        }
    }

    template <typename V, typename Dest>
    void BlockClusterTree::AfFullProduct(ClusterPair pair, const V &v_mid, Dest &result, BCTKernelType kType) const
    {
        for (size_t i = 0; i < pair.cluster1->nElements; i++)
        {
            double a_times_one_i = 0;
            double a_times_v_i = 0;

            size_t f1_ind = pair.cluster1->clusterIndices[i];
            GCFace f1 = mesh->face(f1_ind);
            Vector3 mid1 = faceBarycenters[f1];

            Vector3 n1 = faceNormal(geom, f1);

            double l1 = geom->faceAreas[f1];

            for (size_t j = 0; j < pair.cluster2->nElements; j++)
            {
                size_t f2_ind = pair.cluster2->clusterIndices[j];
                GCFace f2 = mesh->face(f2_ind);
                bool isSame = (f1 == f2);

                Vector3 mid2 = faceBarycenters[f2];
                Vector3 n2 = faceNormal(geom, f2);
                double l2 = geom->faceAreas[f2];

                // Compute the main kernel, times the second mass
                double af_ij = (isSame) ? 0 : l2 * kernel(kType, exp_s, mid1, mid2, n1, n2);

                // We dot this row of Af(i, j) with the all-ones vector, which means we
                // just add up all entries of that row.
                a_times_one_i += af_ij;

                // We also dot it with v_hat(J).
                a_times_v_i += af_ij * v_mid(f2_ind);
            }

            // Multiply in the first mass here
            a_times_one_i *= l1;
            a_times_v_i *= l1;

            // We've computed everything from row i now, so add to the results vector
            double toAdd = 2 * (a_times_one_i * v_mid(f1_ind) - a_times_v_i);
            result(f1_ind) += toAdd;
        }
    }

    template <typename V, typename Dest>
    void BlockClusterTree::AfApproxProduct(ClusterPair pair, const V &v_mid, Dest &result, BCTKernelType kType) const
    {
        Eigen::VectorXd wf_i;
        fillClusterMasses(pair.cluster1, wf_i);
        Eigen::VectorXd wf_j;
        fillClusterMasses(pair.cluster2, wf_j);

        double a_IJ = kernel(kType, exp_s, pair.cluster1->centerOfMass, pair.cluster2->centerOfMass,
                             pair.cluster1->averageNormal, pair.cluster2->averageNormal);

        // Evaluate a(I,J) * w_f(J)^T * 1(J)
        double a_wf_1 = a_IJ * wf_j.sum();

        // Evaluate a(I,J) * w_f(J)^T * v_hat(J)
        double a_wf_J = 0;
        // Dot w_f(J) with v_hat(J)
        for (int j = 0; j < wf_j.rows(); j++)
        {
            a_wf_J += wf_j(j) * v_mid(pair.cluster2->clusterIndices[j]);
        }
        a_wf_J *= a_IJ;

        // Add in the results
        for (int i = 0; i < wf_i.rows(); i++)
        {
            double toAdd = wf_i[i] * 2 * (a_wf_1 * v_mid(pair.cluster1->clusterIndices[i]) - a_wf_J);
            result(pair.cluster1->clusterIndices[i]) += toAdd;
        }
    }

    template <typename V, typename Dest>
    void BlockClusterTree::MultiplyVector(const V &v, Dest &b, BCTKernelType kType, bool addToResult) const
    {
        int nFaces = mesh->nFaces();

        if (kType == BCTKernelType::HighOrder)
        {
            Eigen::SparseMatrix<double> Df = Hs::BuildDfOperator(mesh, geom);

            Eigen::VectorXd v_hat = Df * v;
            size_t nRows = 3 * nFaces;
            // Allocate some space for per-component output
            Eigen::VectorXd b_hat;
            b_hat.setZero(nRows);

            // We now have a vector per vertex, so multiply each component separately
            Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<3>> v_hat_x(v_hat.data(), nFaces);
            Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>> b_hat_x(b_hat.data(), nFaces);
            MultiplyAdmissiblePercolated(v_hat_x, b_hat_x, kType);
            MultiplyInadmissible(v_hat_x, b_hat_x, kType);

            // y component
            Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<3>> v_hat_y(v_hat.data() + 1, nFaces);
            Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>> b_hat_y(b_hat.data() + 1, nFaces);
            MultiplyAdmissiblePercolated(v_hat_y, b_hat_y, kType);
            MultiplyInadmissible(v_hat_y, b_hat_y, kType);

            // z component
            Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<3>> v_hat_z(v_hat.data() + 2, nFaces);
            Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>> b_hat_z(b_hat.data() + 2, nFaces);
            MultiplyAdmissiblePercolated(v_hat_z, b_hat_z, kType);
            MultiplyInadmissible(v_hat_z, b_hat_z, kType);

            if (!addToResult)
                b.setZero();

            b = Df.transpose() * b_hat;
        }
        else
        {
            Eigen::VectorXd v_mid(nFaces);
            v_mid.setZero();

            // Set up input and outputs for metric
            Hs::ApplyMidOperator(mesh, geom, v, v_mid);

            // Set up inputs and outputs for blocks
            Eigen::VectorXd b_mid_adm(nFaces);
            b_mid_adm.setZero();

            // Multiply admissible blocks
            MultiplyAdmissiblePercolated(v_mid, b_mid_adm, kType);
            // Multiply inadmissible blocks
            MultiplyInadmissible(v_mid, b_mid_adm, kType);

            if (!addToResult)
                b.setZero();

            Hs::ApplyMidOperatorTranspose(mesh, geom, b_mid_adm, b);
        }
    }

    template <typename V3, typename Dest>
    void BlockClusterTree::MultiplyVector3(const V3 &v, Dest &b, BCTKernelType kType, bool addToResult) const
    {
        recomputeBarycenters();
        PremultiplyAf1(kType);

        size_t nVerts = mesh->nVertices();
        // Slice the input vector to get every x-coordinate
        Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<3>> v_x(v.data(), nVerts);
        // Slice the output vector to get x-coordinates
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>> dest_x(b.data(), nVerts);
        // Multiply the input x-coords into the output x-coords
        MultiplyVector(v_x, dest_x, kType, addToResult);

        // Same thing for y-coordinates
        Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<3>> v_y(v.data() + 1, nVerts);
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>> dest_y(b.data() + 1, nVerts);
        MultiplyVector(v_y, dest_y, kType, addToResult);

        // Same thing for z-coordinates
        Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<3>> v_z(v.data() + 2, nVerts);
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>> dest_z(b.data() + 2, nVerts);
        MultiplyVector(v_z, dest_z, kType, addToResult);
    }

} // namespace rsurfaces
