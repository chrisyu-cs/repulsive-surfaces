#pragma once

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

        BlockClusterTree(MeshPtr &mesh, GeomPtr &geom, BVHNode6D *root, double sepCoeff, double s_, double e = 0.0);
        ~BlockClusterTree();
        // Loop over all currently inadmissible cluster pairs
        // and subdivide them to their children.
        void splitInadmissibleNodes();
        static bool isPairAdmissible(ClusterPair pair, double coeff);
        static bool isPairSmallEnough(ClusterPair pair);

        void PrintData();
        void PrintAdmissibleClusters(std::ofstream &stream);
        void PrintInadmissibleClusters(std::ofstream &stream);

        void AfFullProduct(ClusterPair pair, const Eigen::VectorXd &v_mid, Eigen::VectorXd &result) const;
        void AfApproxProduct(ClusterPair pair, const Eigen::VectorXd &v_mid, Eigen::VectorXd &result) const;

        // Multiplies v and stores in b. Dispatches to the specific multiplication case below.
        template <typename V, typename Dest>
        void Multiply(V &v, Dest &b) const;

        // Multiplies A * v and stores it in b.
        template <typename V, typename Dest>
        void MultiplyVector(V &v, Dest &b) const;

        // Multiplies A * v, where v holds a vector3 at each vertex in a flattened column,
        //  and stores it in b.
        template <typename V3, typename Dest>
        void MultiplyVector3(V3 &v, Dest &b);

    private:
        // Multiplies the inadmissible clusters for A * v, storing it in b.
        void MultiplyInadmissible(const Eigen::VectorXd &v_hat, Eigen::VectorXd &b_hat) const;
        // Multiplies the admissible clusters for A * v, storing it in b.
        void MultiplyAdmissible(Eigen::VectorXd &v, Eigen::VectorXd &b) const;
        // Multiplies the admissible clusters exactly
        // (without any hierarchical approximation).
        void MultiplyAdmissibleExact(Eigen::VectorXd &v_hat, Eigen::VectorXd &b_hat) const;

        // Multiplies the admissible clusters using a percolation method.
        void MultiplyAdmissiblePercolated(Eigen::VectorXd &v, Eigen::VectorXd &b) const;
        // Multiplies just the kernel matrix A using a percolation method.
        void MultiplyAfPercolated(Eigen::VectorXd &v, Eigen::VectorXd &b) const;

        Eigen::VectorXd Af_1;
        void PremultiplyAf1();

        void fillClusterMasses(BVHNode6D *cluster, Eigen::VectorXd &w) const;
        void OrganizePairsByFirst();

        // Cached list of face barycenters to avoid recomputation
        geometrycentral::surface::FaceData<Vector3> faceBarycenters;
        inline void recomputeBarycenters()
        {
            for (GCFace f : mesh->faces())
            {
                faceBarycenters[f] = faceBarycenter(geom, f);
            }
        }

        double exp_s, separationCoeff;
        double epsilon;
        MeshPtr mesh;
        GeomPtr geom;
        BVHNode6D *tree_root;
        std::vector<ClusterPair> admissiblePairs;
        std::vector<std::vector<ClusterPair>> admissibleByCluster;

        std::vector<ClusterPair> unresolvedPairs;
        std::vector<ClusterPair> inadmissiblePairs;
    };

    template <typename V, typename Dest>
    void BlockClusterTree::Multiply(V &v, Dest &b) const
    {
        MultiplyVector3(v, b);
    }

    template <typename V, typename Dest>
    void BlockClusterTree::MultiplyVector(V &v, Dest &b) const
    {
        int nFaces = mesh->nFaces();
        Eigen::VectorXd v_mid(nFaces);
        v_mid.setZero();

        // Set up input and outputs for metric
        Hs::ApplyMidOperator(mesh, geom, v, v_mid);

        // Set up inputs and outputs for blocks
        Eigen::VectorXd b_mid_adm(nFaces);
        b_mid_adm.setZero();

        // Multiply admissible blocks
        long admStart = currentTimeMilliseconds();
        MultiplyAdmissiblePercolated(v_mid, b_mid_adm);
        long admMid = currentTimeMilliseconds();
        // Multiply inadmissible blocks
        MultiplyInadmissible(v_mid, b_mid_adm);
        long inadmEnd = currentTimeMilliseconds();

        wellSepTime += (admMid - admStart);
        illSepTime += (inadmEnd - admMid);

        b.setZero();
        Hs::ApplyMidOperatorTranspose(mesh, geom, b_mid_adm, b);
    }

    template <typename V3, typename Dest>
    void BlockClusterTree::MultiplyVector3(V3 &v, Dest &b)
    {
        wellSepTime = 0;
        illSepTime = 0;
        traversalTime = 0;

        recomputeBarycenters();

        size_t nVerts = mesh->nVertices();
        // Slice the input vector to get every x-coordinate
        Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<3>> v_x(v.data(), nVerts);
        // Slice the output vector to get x-coordinates
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>> dest_x(b.data(), nVerts);
        // Multiply the input x-coords into the output x-coords
        MultiplyVector(v_x, dest_x);

        // Same thing for y-coordinates
        Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<3>> v_y(v.data() + 1, nVerts);
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>> dest_y(b.data() + 1, nVerts);
        MultiplyVector(v_y, dest_y);

        // Same thing for z-coordinates
        Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<3>> v_z(v.data() + 2, nVerts);
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<3>> dest_z(b.data() + 2, nVerts);
        MultiplyVector(v_z, dest_z);

        std::cout << "      * Admissible blocks time = " << wellSepTime << " ms" << std::endl;
        std::cout << "      * Inadmissible blocks time = " << illSepTime << " ms" << std::endl;
        std::cout << "      * Weight copying time = " << traversalTime << " ms" << std::endl;
    }

} // namespace rsurfaces
