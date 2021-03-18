    #pragma once

#include "bct_kernel_type.h"
#include "optimized_cluster_tree.h"
#include "interaction_data.h"

namespace rsurfaces
{

    class OptimizedBlockClusterTree
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

        OptimizedBlockClusterTree(OptimizedClusterTree* S_, OptimizedClusterTree* T_, const mreal alpha_, const mreal beta_, const mreal theta_, bool exploit_symmetry_ = true, bool upper_triangular_ = false);

        ~OptimizedBlockClusterTree()
        {   
            mreal_free(hi_diag);
            mreal_free(lo_diag);
            mreal_free(fr_diag);
        };

        mutable OptimizedClusterTree* S; // "left" OptimizedClusterTree (output side of matrix-vector multiplication)
        mutable OptimizedClusterTree* T; // "right" OptimizedClusterTree (input side of matrix-vector multiplication)

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


        // Scaling parameters for the matrix-vector product.
        // E.g. one can perform  u = hi_factor * A_hi * v. With MKL, this comes at no cost, because it is fused into the matrix multiplications anyways.
        mreal hi_far_factor = 1.;
        mreal lo_far_factor = 1.;
        mreal fr_far_factor = 1.;
        
        mreal hi_near_factor = 1.;
        mreal lo_near_factor = 1.;
        mreal fr_near_factor = 1.;

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
        
        template<typename OptBCTPtr>
        void AddObstacleCorrection(OptBCTPtr bct12)
        {
            // Suppose that bct11 = this;
            // The joint bct of the union of mesh1 and mesh2 can be written in block matrix for as
            //  bct = {
            //            { bct11, bct12 },
            //            { bct21, bct22 }
            //        },
            // where bct11 and bct22 are the instances of OptimizedBlockClusterTree of mesh1 and mesh2, respectively, bct12 is cross interaction OptimizedBlockClusterTree of mesh1 and mesh2, and bct21 is the transpose of bct12.
            // However, the according matrix (on the space of dofs on the primitives) would be
            //  A   = {
            //            { A11 + diag( A12 * one2 ) , A12                      },
            //            { A21                      , A22 + diag( A21 * one1 ) }
            //        },
            // where one1 and one2 are all-1-vectors on the primitives of mesh1 and mesh2, respectively.
            // OptimizedBlockClusterTree::AddObstacleCorrection is supposed to compute diag( A12 * one2 ) and to add it to the diagonal of A11.
            // Then the bct11->Multiply will also multiply with the obstacle.
            
            if( (S == T) && (T == bct12->S) )
            {
                RequireMetrics();
                bct12->RequireMetrics();

                if( fr_factor != bct12->fr_factor )
                {
                    wprint("AddToDiagonal: The values of fr_factor of the two instances of OptimizedBlockClusterTree do not coincide.");
                }
                if( hi_factor != bct12->hi_factor )
                {
                    wprint("AddToDiagonal: The values of hi_factor of the two instances of OptimizedBlockClusterTree do not coincide.");
                }
                if( lo_factor != bct12->lo_factor )
                {
                    wprint("AddToDiagonal: The values of lo_factor of the two instances of OptimizedBlockClusterTree do not coincide.");
                }
                
                mint n = T->primitive_count;
                
                mreal * restrict const fr_target = fr_diag;
                mreal * restrict const hi_target = hi_diag;
                mreal * restrict const lo_target = lo_diag;
                
                mreal const * restrict const fr_source = bct12->fr_diag;
                mreal const * restrict const hi_source = bct12->hi_diag;
                mreal const * restrict const lo_source = bct12->lo_diag;
                
                #pragma omp parallel for simd aligned( fr_target, hi_target, lo_target, fr_source, hi_source, lo_source : ALIGN )
                for( mint i = 0; i < n; ++ i)
                {
                    fr_target[i] += fr_source[i];
                    hi_target[i] += hi_source[i];
                    lo_target[i] += lo_source[i];
                }
            }
            else
            {
                if( S != T )
                {
                    eprint("AddToDiagonal: Instance of OptimizedBlockClusterTree is not symmetric. Doing nothing.");
                }
                if( S != bct12->S )
                {
                    eprint("AddToDiagonal: The two instances of OptimizedBlockClusterTree are not compatible. Doing nothing.");
                }
            }
        }
        
        void PrintStats(){
            std::cout << "\n==== OptimizedBlockClusterTree Stats ====" << std::endl;
            
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
            
            std::cout << "==== OptimizedBlockClusterTree Stats ====\n" << std::endl;
            
        };

    }; //OptimizedBlockClusterTree

} // namespace rsurfaces
