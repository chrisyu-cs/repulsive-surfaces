#pragma once

#include "bct_kernel_type.h"
#include "cluster_tree2.h"
#include "interaction_data.h"

namespace rsurfaces
{

    class BlockClusterTree2
    {
    public:
        //Main interface routines first:

        void Multiply(Eigen::VectorXd &input, Eigen::VectorXd &output, const mint k, BCTKernelType type, bool addToResult = false) const; // <--- Main interface routine for Chris.

        void Multiply(Eigen::VectorXd &input, Eigen::VectorXd &output, BCTKernelType type, bool addToResult = false) const; // <--- Main interface routine for Chris.

        void Multiply(Eigen::MatrixXd &input, Eigen::MatrixXd &output, BCTKernelType type, bool addToResult = false) const; // <--- Main interface routine for Chris.

        template <typename V3, typename Dest>
        void MultiplyV3(const V3 &input, Dest &output, BCTKernelType type, bool addToResult = false) const
        {
            Eigen::VectorXd inVec = input;
            Eigen::VectorXd outVec;
            outVec.setZero(output.rows());
            Multiply(inVec, outVec, 3, type, addToResult);
            if (addToResult)
            {
                output += outVec;
            }
            else
            {
                output = outVec;
            }
        }

        bool disableNearField = false;

        BlockClusterTree2(std::shared_ptr<ClusterTree2> S_, std::shared_ptr<ClusterTree2> T_, const mreal alpha_, const mreal beta_, const mreal theta_, bool exploit_symmetry_ = true, bool upper_triangular_ = false);

        ~BlockClusterTree2()
        {
//            // If the two pointers are distinct, delete both
//            if (S != T)
//            {
//                if (S)
//                    delete S;
//                if (T)
//                    delete T;
//            }
//            // If they're the same, just delete one
//            else
//            {
//                if (S)
//                    delete S;
//            }
            
            mreal_free(hi_diag);
            mreal_free(lo_diag);
            mreal_free(fr_diag);
        };

        mutable std::shared_ptr<ClusterTree2> S; // "left" ClusterTree2 (output side of matrix-vector multiplication)
        mutable std::shared_ptr<ClusterTree2> T; // "right" ClusterTree2 (input side of matrix-vector multiplication)

        mint dim = 3;
        mreal theta2 = 0.25;
        mint thread_count = 1;
        mint tree_thread_count = 1;
        mreal alpha = 6.0;
        mreal beta = 12.0;
        mreal exp_s = 2.0 - 1.0 / 3.0; // differentiability of the energy space
        const mint intrinsic_dim = 2;  // Only for surfaces at the moment. It would be intrinsic_dim = 1 for curves.

        mreal hi_exponent = -0.5 * (2.0 * (2.0 / 3.0) + 2.0); // The only exponent we have to use for pow to compute matrix entries. All other exponents have been optimized away.
                                                              //    mreal fr_exponent;

        // TODO: better use a hash table with keys from BCTKernelType to store factor, diag, and stuff. Easier to access, more easily to extend, and way easier to loop over these three cases.

        // Scaling parameters for the matrix-vector product.
        // E.g. one can perform  u = hi_factor * A_hi * v. With MKL, this comes at no cost, because it is fused into the matrix multiplications anyways.
        mreal hi_factor = 1.;
        mreal lo_factor = 1.;
        mreal fr_factor = 1.;

        // Product of the kernel matrix with the constant-1-vector.
        // Need to be updated if hi_factor, lo_factor, or fr_factor are changed!
        // Assumed to be in EXTERNAL ORDERING!
        mutable  mreal * restrict hi_diag = NULL;
        mutable  mreal * restrict lo_diag = NULL;
        mutable  mreal * restrict fr_diag = NULL;
        
        // TODO: Maybe these "diag" - vectors should become members to S and T?
        // Remark: If S != T, the "diags" are not used.

        bool metrics_initialized = false;
        bool is_symmetric = false;
        bool exploit_symmetry = false;
        bool upper_triangular = false;
        
        // If exploit_symmetry != 1, S == T is assume and only roughly half the block clusters are generated during the split pass performed by CreateBlockClusters.
        // If upper_triangular != 0 and if exploit_symmetry != 0, only the upper triangle of the interaction matrices will be generated. --> CreateBlockClusters will be faster.
        // If exploit_symmetry 1= 1 and upper_triangular 1= 0 then the block cluster twins are generated _at the end_ of the splitting pass by CreateBlockClusters.

        std::shared_ptr<InteractionData> far;  // far and near are data containers for far and near field, respectively.
        std::shared_ptr<InteractionData> near; // They also perform the matrix-vector products.

        mreal FarFieldEnergy0();
        mreal DFarFieldEnergy0Helper();
        mreal NearFieldEnergy0();
        mreal DNearFieldEnergy0Helper();
        
        mreal FarFieldEnergyInteger0();
        mreal DFarFieldEnergyInteger0Helper();
        mreal NearFieldEnergyInteger0();
        mreal DNearFieldEnergyInteger0Helper();
        
        mreal BarnesHutEnergy0();
        mreal DBarnesHutEnergy0Helper();
        
        // TODO: Transpose operation
        //    void MultiplyTransposed( const mreal * const restrict P_input, mreal * const restrict P_output, const mint  cols, BCTKernelType type, bool addToResult = false );
        //
        //    void MultiplyTransposed( Eigen::MatrixXd &input, Eigen::MatrixXd &output, BCTKernelType type, bool addToResult = false );
        //
        //    void MultiplyTransposed( Eigen::VectorXd &input, Eigen::VectorXd &output, BCTKernelType type, bool addToResult = false );

        //private:  // made public only for debugging

        void CreateBlockClusters(); // Creates InteractionData far and near for far and near field, respectively.

        void SplitBlockCluster(
            A_Vector<A_Deque<mint>> &sep_i,  //  +
            A_Vector<A_Deque<mint>> &sep_j,  //  |   <-- separate containers for each thread
            A_Vector<A_Deque<mint>> &nsep_i, //  |
            A_Vector<A_Deque<mint>> &nsep_j, //  +
            const mint i,                    //  <-- index of first  cluster in the block cluster
            const mint j,                    //  <-- index of second cluster in the block cluster
            const mint free_thread_count     //  <-- helps to manage task creation
        );

        void RequireMetrics();
        
        void FarFieldInteraction(); // Compute nonzero values of sparse far field interaction matrices.

        void NearFieldInteractionCSR(); // Compute nonzero values of sparse near field interaction matrices in CSR format.

        void InternalMultiply(BCTKernelType type) const;

        void ComputeDiagonals();
        
        void AddObstacleCorrection( BlockClusterTree2 * bct12);
        
        void PrintStats(){
            std::cout << "\n==== BlockClusterTree2 Stats ====" << std::endl;
            
            std::cout << " dim                 = " <<  dim << std::endl;
            std::cout << " theta               = " <<  sqrt(theta2) << std::endl;
            std::cout << " thread_count        = " <<  thread_count << std::endl;
            std::cout << " tree_thread_count   = " <<  tree_thread_count << std::endl;
            
            std::cout << " S->cluster_count    = " <<  S->cluster_count << std::endl;
            std::cout << " T->cluster_count    = " <<  T->cluster_count << std::endl;
            std::cout << " separated blocks    = " <<  far->nnz << std::endl;
            std::cout << " nonseparated blocks = " <<  near->b_nnz << std::endl;
            
            std::cout << "\n---- bool data ----" << std::endl;
            
            std::cout << " metrics_initialized = " <<  metrics_initialized << std::endl;
            std::cout << " is_symmetric        = " <<  is_symmetric << std::endl;
            std::cout << " exploit_symmetry    = " <<  exploit_symmetry << std::endl;
            std::cout << " upper_triangular    = " <<  upper_triangular << std::endl;
//
//            std::cout << "\n---- double data ----" << std::endl;
//
//            std::cout << " alpha       = " <<  alpha << std::endl;
//            std::cout << " beta        = " <<  beta << std::endl;
//            std::cout << " exp_s       = " <<  exp_s << std::endl;
//            std::cout << " hi_exponent = " <<  hi_exponent << std::endl;
//            std::cout << " hi_factor   = " <<  hi_factor << std::endl;
//            std::cout << " lo_factor   = " <<  lo_factor << std::endl;
//            std::cout << " fr_factor   = " <<  fr_factor << std::endl;
            
            std::cout << "==== BlockClusterTree2 Stats ====\n" << std::endl;
            
        };

    }; //BlockClusterTree2

} // namespace rsurfaces
