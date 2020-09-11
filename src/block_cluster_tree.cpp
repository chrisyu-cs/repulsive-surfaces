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

    BlockClusterTree::BlockClusterTree(MeshPtr mesh_, GeomPtr geom_, BVHNode6D *root, double sepCoeff, double s_, double e)
    {
        mesh = mesh_;
        geom = geom_;
        exp_s = s_;
        separationCoeff = sepCoeff;
        epsilon = e;

        // std::cout << "Using " << nThreads << " threads." << std::endl;

        tree_root = root;
        ClusterPair pair{tree_root, tree_root, 0};
        unresolvedPairs.push_back(pair);

        int depth = 0;
        while (unresolvedPairs.size() > 0)
        {
            splitInadmissibleNodes(depth);
            depth++;
        }

        PremultiplyAf1();
    }

    BlockClusterTree::~BlockClusterTree()
    {
        //delete threadpool;
    }

    void BlockClusterTree::splitInadmissibleNodes(int depth)
    {
        std::vector<ClusterPair> nextPairs;

        for (ClusterPair pair : unresolvedPairs)
        {
            pair.depth = depth;
            if (pair.cluster1->NumElements() == 0 || pair.cluster2->NumElements() == 0)
            {
                // Drop pairs where one of the sides has 0 vertices
                continue;
            }
            else if (pair.cluster1->NumElements() == 1 && pair.cluster2->NumElements() == 1)
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
                for (size_t i = 0; i < pair.cluster1->children.size(); i++)
                {
                    for (size_t j = 0; j < pair.cluster2->children.size(); j++)
                    {
                        ClusterPair pair_ij{pair.cluster1->children[i], pair.cluster2->children[j], depth + 1};
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
        int s1 = pair.cluster1->NumElements();
        int s2 = pair.cluster2->NumElements();
        return (s1 <= 1) || (s2 <= 1) || (s1 + s2 <= 8);
    }

    bool BlockClusterTree::isPairAdmissible(ClusterPair pair, double theta)
    {
        // A cluster is never admissible with itself
        if (pair.cluster1 == pair.cluster2)
            return false;
        // Compute distance between centers of masses of clusters, along with cluster bounding radii
        double distance = norm(pair.cluster1->centerOfMass - pair.cluster2->centerOfMass);
        double radius1 = norm(pair.cluster1->maxCoords - pair.cluster1->minCoords) / 2;
        double radius2 = norm(pair.cluster2->maxCoords - pair.cluster2->minCoords) / 2;
        // A cluster is never admissible with a cluster whose center is inside its bounding radius
        if (distance < radius1 || distance < radius2)
            return false;

        // Compute Barnes-Hut distance ratios from boundary to boundary of bounding radii
        double ratio1 = pair.cluster1->nodeRatio(distance - radius2);
        double ratio2 = pair.cluster2->nodeRatio(distance - radius1);

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

    void BlockClusterTree::MultiplyInadmissible(const Eigen::VectorXd &v_hat, Eigen::VectorXd &b_hat) const
    {
        Eigen::VectorXd result;
        result.setZero(b_hat.rows());

        for (ClusterPair const &pair : inadmissiblePairs)
        {
            AfFullProduct(pair, v_hat, result);
        }

        b_hat += result;
    }

    void BlockClusterTree::PremultiplyAf1()
    {
        Af_1.setOnes(tree_root->NumElements());
        MultiplyAfPercolated(Af_1, Af_1);
    }

    void BlockClusterTree::MultiplyAdmissiblePercolated(Eigen::VectorXd &v, Eigen::VectorXd &b) const
    {
        MultiplyAfPercolated(v, b);
        b = 2 * (Af_1.asDiagonal() * v - b);
    }

    // Get the dot product W^T * J for the given node in a data tree (and all children)
    void percolateWtDot(DataTree<PercolationData> *dataTree, Eigen::VectorXd &v, MeshPtr mesh, GeomPtr geom)
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

    void percolateJ(DataTree<PercolationData> *dataTree, double parentB, Eigen::VectorXd &b)
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

    void BlockClusterTree::MultiplyAfPercolated(Eigen::VectorXd &v, Eigen::VectorXd &b) const
    {
        Eigen::VectorXd w(tree_root->NumElements());
        fillClusterMasses(tree_root, w);

        DataTreeContainer<PercolationData> *treeContainer = tree_root->CreateDataTree<PercolationData>();
        // Percolate W^T * v upward through the tree
        percolateWtDot(treeContainer->tree, v, mesh, geom);

        // For each cluster I, we need to sum over all clusters J that are
        // admissible with it. Since we already have a list of all admissible
        // pairs, we can just do this for all clusters at once.
        for (ClusterPair const &pair : admissiblePairs)
        {
            double a_IJ = Hs::MetricDistanceTermFrac(exp_s, pair.cluster1->centerOfMass, pair.cluster2->centerOfMass);
            DataTree<PercolationData> *data_I = treeContainer->byIndex[pair.cluster1->nodeID];
            DataTree<PercolationData> *data_J = treeContainer->byIndex[pair.cluster2->nodeID];
            // Each I gets a sum of a_IJ * V_J for all admissible J
            data_I->data.B += a_IJ * data_J->data.wtDot;
        }

        // Percolate downward from the root
        percolateJ(treeContainer->tree, 0, b);

        // Now the result is stored in b, so just clean up
        delete treeContainer;
    }

    void BlockClusterTree::AfFullProduct(ClusterPair pair, const Eigen::VectorXd &v_mid, Eigen::VectorXd &result) const
    {
        std::vector<double> a_times_one(pair.cluster1->NumElements());
        std::vector<double> a_times_v(pair.cluster1->NumElements());

        for (size_t i = 0; i < pair.cluster1->NumElements(); i++)
        {
            int f1_ind = pair.cluster1->clusterIndices[i];
            GCFace f1 = mesh->face(f1_ind);
            Vector3 mid1 = faceBarycenter(geom, f1);
            double l1 = geom->faceArea(f1);

            for (size_t j = 0; j < pair.cluster2->NumElements(); j++)
            {
                int f2_ind = pair.cluster2->clusterIndices[j];
                GCFace f2 = mesh->face(f2_ind);
                bool isSame = (f1 == f2);

                Vector3 mid2 = faceBarycenter(geom, f2);
                double l2 = geom->faceArea(f2);

                // Save on a few operations by only multiplying l2 now,
                // and multiplying l1 only once, after inner loop
                double distTerm = Hs::MetricDistanceTermFrac(exp_s, mid1, mid2);

                double af_ij = (isSame) ? 0 : l2 * distTerm;

                // We dot this row of Af(i, j) with the all-ones vector, which means we
                // just add up all entries of that row.
                a_times_one[i] += af_ij;

                // We also dot it with v_hat(J).
                a_times_v[i] += af_ij * v_mid(f2_ind);
            }

            a_times_one[i] *= l1;
            a_times_v[i] *= l1;

            // We've computed everything from row i now, so add to the results vector
            double toAdd = 2 * (a_times_one[i] * v_mid(f1_ind) - a_times_v[i]);
            result(f1_ind) += toAdd;
        }
    }

    void BlockClusterTree::fillClusterMasses(BVHNode6D *cluster, Eigen::VectorXd &w) const
    {
        int nElts = cluster->NumElements();
        w.setZero(nElts);
        for (size_t i = 0; i < cluster->NumElements(); i++)
        {
            w(i) = geom->faceArea(mesh->face(cluster->clusterIndices[i]));
        }
    }

    void BlockClusterTree::AfApproxProduct(ClusterPair pair, const Eigen::VectorXd &v_mid, Eigen::VectorXd &result) const
    {
        Eigen::VectorXd wf_i;
        fillClusterMasses(pair.cluster1, wf_i);
        Eigen::VectorXd wf_j;
        fillClusterMasses(pair.cluster2, wf_j);

        double a_IJ = Hs::MetricDistanceTermFrac(exp_s, pair.cluster1->centerOfMass, pair.cluster2->centerOfMass);

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
} // namespace rsurfaces
