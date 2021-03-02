#include "block_cluster_tree.h"

#include <fstream>
#include <sstream>
#include <queue>
#include <map>

#include "helpers.h"

namespace rsurfaces
{
    long BlockClusterTree::illSepTime = 0;
    long BlockClusterTree::wellSepTime = 0;
    long BlockClusterTree::traversalTime = 0;

    BlockClusterTree::BlockClusterTree(const MeshPtr &mesh_, const GeomPtr &geom_, BVHNode6D *root, double sepCoeff, double s_, double e)
        : faceBarycenters(*mesh_), mesh(mesh_), geom(geom_)
    {
        exp_s = s_;
        separationCoeff = 2 * sepCoeff;
        epsilon = e;

        // std::cout << "Using " << nThreads << " threads." << std::endl;

        tree_root = root;
        ClusterPair pair{tree_root, tree_root};
        unresolvedPairs.push_back(pair);

        while (unresolvedPairs.size() > 0)
        {
            splitInadmissibleNodes();
        }

        // Need to sort the pairs before doing any multiplication
        OrganizePairsByFirst();

        highInitialized = false;
        lowInitialized = false;
        fracInitialized = false;
    }

    BlockClusterTree::~BlockClusterTree()
    {
        //delete threadpool;
    }

    void BlockClusterTree::splitInadmissibleNodes()
    {
        std::vector<ClusterPair> nextPairs;

        for (ClusterPair pair : unresolvedPairs)
        {
            if (pair.cluster1->nElements == 0 || pair.cluster2->nElements == 0)
            {
                // Drop pairs where one of the sides has 0 vertices
                continue;
            }
            else if (pair.cluster1->nElements == 1 && pair.cluster2->nElements == 1)
            {
                // If this is two singleton vertices, put in the inadmissible list
                // so they get multiplied accurately
                inadmissiblePairs.push_back(pair);
            }
            else if (isPairAdmissible(pair, separationCoeff))
            {
                // If the pair is admissible, mark it as such and leave it
                admissiblePairs.push_back(pair);
            }
            else if (isPairSmallEnough(pair))
            {
                inadmissiblePairs.push_back(pair);
            }
            else
            {
                // Otherwise, subdivide it into child pairs
                for (size_t i = 0; i < BVH_N_CHILDREN; i++)
                {
                    for (size_t j = 0; j < BVH_N_CHILDREN; j++)
                    {
                        ClusterPair pair_ij{pair.cluster1->children[i], pair.cluster2->children[j]};
                        nextPairs.push_back(pair_ij);
                    }
                }
            }
        }
        // Replace the inadmissible pairs by the next set
        unresolvedPairs.clear();
        unresolvedPairs = nextPairs;
    }

    bool BlockClusterTree::isPairSmallEnough(ClusterPair pair)
    {
        int s1 = pair.cluster1->nElements;
        int s2 = pair.cluster2->nElements;
        return (s1 <= 1) || (s2 <= 1) || (s1 + s2 <= 8);
    }

    bool BlockClusterTree::isPairAdmissible(ClusterPair pair, double theta)
    {
        // A cluster is never admissible with itself
        if (pair.cluster1 == pair.cluster2)
            return false;

        // A cluster is never admissible with a cluster whose center is inside its bounding box
        if (pair.cluster1->boxContainsPoint(pair.cluster2->centerOfMass) ||
            pair.cluster2->boxContainsPoint(pair.cluster1->centerOfMass))
        {
            return false;
        }

        // Compute distance between centers of masses of clusters, along with cluster bounding radii
        double distance = norm(pair.cluster1->centerOfMass - pair.cluster2->centerOfMass);

        // Compute Barnes-Hut distance ratios
        double ratio1 = pair.cluster1->nodeRatio(distance);
        double ratio2 = pair.cluster2->nodeRatio(distance);

        // Consider admissible only if both Barnes-Hut checks pass
        bool isAdm = fmax(ratio1, ratio2) < theta;
        return isAdm;
    }

    void BlockClusterTree::PrintData()
    {
        std::cout << admissiblePairs.size() << " admissible pairs" << std::endl;
        std::cout << inadmissiblePairs.size() << " inadmissible pairs" << std::endl;
    }

    void BlockClusterTree::PrintAdmissibleClusters(std::ofstream &stream)
    {
        for (ClusterPair p : admissiblePairs)
        {
            stream << p.cluster1->nodeID << ", " << p.cluster2->nodeID << std::endl;
        }
    }

    void BlockClusterTree::PrintInadmissibleClusters(std::ofstream &stream)
    {
        for (ClusterPair p : inadmissiblePairs)
        {
            stream << p.cluster1->nodeID << ", " << p.cluster2->nodeID << std::endl;
        }
    }

    void BlockClusterTree::PremultiplyAf1(BCTKernelType kType) const
    {
        switch (kType)
        {
        case BCTKernelType::FractionalOnly:
            if (!fracInitialized)
            {
                Af_1_Frac.setOnes(tree_root->nElements);
                MultiplyAfPercolated(Af_1_Frac, Af_1_Frac, kType);
                fracInitialized = true;
            }
            break;
        case BCTKernelType::HighOrder:
            if (!highInitialized)
            {
                Af_1_High.setOnes(tree_root->nElements);
                MultiplyAfPercolated(Af_1_High, Af_1_High, kType);
                highInitialized = true;
            }
            break;
        case BCTKernelType::LowOrder:
            if (!lowInitialized)
            {
                Af_1_Low.setOnes(tree_root->nElements);
                MultiplyAfPercolated(Af_1_Low, Af_1_Low, kType);
                lowInitialized = true;
            }
            break;
        default:
            throw std::runtime_error("Unknown kernel type in PremultiplyAf1.");
        }
    }

    const Eigen::VectorXd &BlockClusterTree::getPremultipliedAf1(BCTKernelType kType) const
    {
        switch (kType)
        {
        case BCTKernelType::FractionalOnly:
            if (!fracInitialized)
                throw std::runtime_error("You need to manually call PremultiplyAf1 for FractionalOnly.");
            return Af_1_Frac;
        case BCTKernelType::HighOrder:
            if (!highInitialized)
                throw std::runtime_error("You need to manually call PremultiplyAf1 for HighOrder.");
            return Af_1_High;
        case BCTKernelType::LowOrder:
            if (!lowInitialized)
                throw std::runtime_error("You need to manually call PremultiplyAf1 for LowOrder.");
            return Af_1_Low;
        default:
            throw std::runtime_error("Unknown kernel type in getPremultipliedAf1.");
        }
    }

    void BlockClusterTree::OrganizePairsByFirst()
    {
        // Bucket cluster pairs by which one occurs in the first position
        admissibleByCluster.clear();
        admissibleByCluster.resize(tree_root->numNodesInBranch);

        for (ClusterPair const &pair : admissiblePairs)
        {
            admissibleByCluster[pair.cluster1->nodeID].push_back(pair);
        }
    }

    void BlockClusterTree::fillClusterMasses(BVHNode6D *cluster, Eigen::VectorXd &w) const
    {
        int nElts = cluster->nElements;
        w.setZero(nElts);
        for (size_t i = 0; i < cluster->nElements; i++)
        {
            w(i) = geom->faceAreas[mesh->face(cluster->clusterIndices[i])];
        }
    }
} // namespace rsurfaces
