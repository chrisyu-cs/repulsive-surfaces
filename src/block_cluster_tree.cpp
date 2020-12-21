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

    BlockClusterTree::BlockClusterTree(MeshPtr &mesh_, GeomPtr &geom_, BVHNode6D *root, double sepCoeff, double s_, double e)
        : faceBarycenters(*mesh_)
    {
        mesh = mesh_;
        geom = geom_;
        exp_s = s_;
        separationCoeff = sepCoeff;
        epsilon = e;
        kernelType = BCTKernelType::FractionalOnly;

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
        // Now we can multiply
        PremultiplyAf1();
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

    void BlockClusterTree::MultiplyAdmissible(Eigen::VectorXd &v_hat, Eigen::VectorXd &b_hat) const
    {
        for (ClusterPair const &pair : admissiblePairs)
        {
            AfApproxProduct(pair, v_hat, b_hat);
        }
    }

    void BlockClusterTree::MultiplyAdmissibleExact(Eigen::VectorXd &v_hat, Eigen::VectorXd &b_hat) const
    {
        for (ClusterPair const &pair : admissiblePairs)
        {
            AfFullProduct(pair, v_hat, b_hat);
        }
    }


    void BlockClusterTree::PremultiplyAf1()
    {
        Af_1.setOnes(tree_root->nElements);
        MultiplyAfPercolated(Af_1, Af_1);
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
