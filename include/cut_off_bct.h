    #pragma once

#include "bct_kernel_type.h"
#include "optimized_cluster_tree.h"
#include "optimized_bct.h"
#include "interaction_data.h"

namespace rsurfaces
{

    
    class CutOffBlockClusterTree
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

        CutOffBlockClusterTree( OptimizedClusterTree* S_, OptimizedClusterTree* T_, const mreal alpha_, const mreal beta_, const mreal theta_,
                                   mreal weight_ = 1.,
                                   BCTSettings settings_ = BCTDefaultSettings );

        ~CutOffBlockClusterTree()
        {
            ptic("~CutOffBlockClusterTree");
            #pragma omp parallel
            {
                #pragma omp single
                {
                    #pragma omp task
                    {
                        safe_free(hi_diag);
                    }
                    #pragma omp task
                    {
                        safe_free(lo_diag);
                    }
                    #pragma omp task
                    {
                        safe_free(fr_diag);
                    }
                    #pragma omp taskwait
                }
            }
            ptoc("~CutOffBlockClusterTree");
        };

        mutable OptimizedClusterTree* S; // "left" OptimizedClusterTree (output side of matrix-vector multiplication)
        mutable OptimizedClusterTree* T; // "right" OptimizedClusterTree (input side of matrix-vector multiplication)

        mint dim = 3;
        mreal squared_cut_off = 1.;
        mreal squared_inv_cut_off = 1.;
        mint thread_count = 1;
        mint tree_thread_count = 1;
        mreal alpha = 6.0;
        mreal beta = 12.0;
        mreal exp_s = 2.0 - 1.0 / 3.0; // differentiability of the energy space
        const mint intrinsic_dim = 2;  // Only for surfaces at the moment. It would be intrinsic_dim = 1 for curves.

        mreal hi_exponent = -0.5 * (2.0 * (2.0 / 3.0) + 2.0); // The only exponent we have to use for pow to compute matrix entries. All other exponents have been optimized away.
                                                              //    mreal fr_exponent;
        mreal weight = 1.;

        // Product of the kernel matrix with the constant-1-vector.
        // Need to be updated if hi_factor, lo_factor, or fr_factor are changed!
        // Assumed to be in EXTERNAL ORDERING!
        mutable  mreal * restrict hi_diag = NULL;
        mutable  mreal * restrict lo_diag = NULL;
        mutable  mreal * restrict fr_diag = NULL;
        
        // TODO: Maybe these "diag" - vectors should become members to S and T?
        // Remark: If S != T, the "diags" are not used.

        BCTSettings settings;
        
        bool block_clusters_initialized = false;
        bool metrics_initialized = false;
        bool is_symmetric = false;
//        std::shared_ptr<InteractionData> far;  // far and near are data containers for far and near field, respectively.
        std::shared_ptr<InteractionData> near; // They also perform the matrix-vector products.

        //private:  // made public only for debugging

        void RequireBlockClusters(); // Creates InteractionData far and near for far and near field, respectively.

        void SplitBlockCluster(
            A_Vector<A_Deque<mint>> &nsep_i,
            A_Vector<A_Deque<mint>> &nsep_j,
            const mint i,                    //  <-- index of first  cluster in the block cluster
            const mint j,                    //  <-- index of second cluster in the block cluster
            const mint free_thread_count     //  <-- helps to manage task creation
        );

        void RequireMetrics();
        
//        void FarFieldInteraction(); // Compute nonzero values of sparse far field interaction matrices.

        void NearFieldInteraction_CSR(); // Compute nonzero values of sparse near field interaction matrices in CSR format.format.
        
        void InternalMultiply(BCTKernelType type) const;

        void ComputeDiagonals();
        
        template<typename CutOffBCTPtr>
        void AddObstacleCorrection(CutOffBCTPtr bct12)
        {
            ptic("CutOffBlockClusterTree::AddObstacleCorrection");
            // Suppose that bct11 = this;
            // The joint bct of the union of mesh1 and mesh2 can be written in block matrix for as
            //  bct = {
            //            { bct11, bct12 },
            //            { bct21, bct22 }
            //        },
            // where bct11 and bct22 are the instances of CutOffBlockClusterTree of mesh1 and mesh2, respectively, bct12 is cross interaction CutOffBlockClusterTree of mesh1 and mesh2, and bct21 is the transpose of bct12.
            // However, the according matrix (on the space of dofs on the primitives) would be
            //  A   = {
            //            { A11 + diag( A12 * one2 ) , A12                      },
            //            { A21                      , A22 + diag( A21 * one1 ) }
            //        },
            // where one1 and one2 are all-1-vectors on the primitives of mesh1 and mesh2, respectively.
            // CutOffBlockClusterTree::AddObstacleCorrection is supposed to compute diag( A12 * one2 ) and to add it to the diagonal of A11.
            // Then the bct11->Multiply will also multiply with the obstacle.
            
            if( (S == T) && (T == bct12->S) )
            {
                RequireMetrics();
                bct12->RequireMetrics();

//                if( far->fr_factor != bct12->far->fr_factor )
//                {
//                    wprint("AddObstacleCorrection: The values of far->fr_factor of the two instances of CutOffBlockClusterTree do not coincide.");
//                }
//                if( far->hi_factor != bct12->far->hi_factor )
//                {
//                    wprint("AddObstacleCorrection: The values of far->hi_factor of the two instances of CutOffBlockClusterTree do not coincide.");
//                }
//                if( far->lo_factor != bct12->far->lo_factor )
//                {
//                    wprint("AddObstacleCorrection: The values of far->lo_factor of the two instances of CutOffBlockClusterTree do not coincide.");
//                }
//                if( near->fr_factor != bct12->near->fr_factor )
//                {
//                    wprint("AddObstacleCorrection: The values of near->fr_factor of the two instances of CutOffBlockClusterTree do not coincide.");
//                }
//                if( near->hi_factor != bct12->near->hi_factor )
//                {
//                    wprint("AddObstacleCorrection: The values of near->hi_factor of the two instances of CutOffBlockClusterTree do not coincide.");
//                }
//                if( near->lo_factor != bct12->near->lo_factor )
//                {
//                    wprint("AddObstacleCorrection: The values of near->lo_factor of the two instances of CutOffBlockClusterTree do not coincide.");
//                }
                
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
                    eprint("AddToDiagonal: Instance of CutOffBlockClusterTree is not symmetric. Doing nothing.");
                }
                if( S != bct12->S )
                {
                    eprint("AddToDiagonal: The two instances of CutOffBlockClusterTree are not compatible. Doing nothing.");
                }
            }
            ptoc("CutOffBlockClusterTree::AddObstacleCorrection");
        }
        
        void PrintStats(){
            std::cout << "\n==== CutOffBlockClusterTree Stats ====" << std::endl;
            
            std::cout << " dim                 = " <<  dim << std::endl;
            std::cout << " cut_off               = " << sqrt(squared_cut_off) << std::endl;
            std::cout << " thread_count        = " <<  thread_count << std::endl;
            std::cout << " tree_thread_count   = " <<  tree_thread_count << std::endl;
            
            std::cout << " S->cluster_count    = " <<  S->cluster_count << std::endl;
            std::cout << " T->cluster_count    = " <<  T->cluster_count << std::endl;
            std::cout << " separated blocks    = " <<  far->nnz << std::endl;
            std::cout << " nonseparated blocks = " <<  near->b_nnz << std::endl;
            
            std::cout << "\n---- bool data ----" << std::endl;
            
            std::cout << " metrics_initialized = " <<  metrics_initialized << std::endl;
            std::cout << " is_symmetric        = " <<  is_symmetric << std::endl;
            std::cout << " exploit_symmetry    = " <<  settings.exploit_symmetry << std::endl;
            std::cout << " upper_triangular    = " <<  settings.upper_triangular << std::endl;
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
            
            std::cout << "==== CutOffBlockClusterTree Stats ====\n" << std::endl;
            
        };
    

    
        #pragma omp declare simd
        inline void ComputeInteraction( mreal x1, mreal x2, mreal x3, mreal n1, mreal n2, mreal n3,
                                               mreal y1, mreal y2, mreal y3, mreal m1, mreal m2, mreal m3,
                                               mreal t1, mreal t2, mreal hi_exponent,
                                               mreal & fr_val, mreal & lo_val, mreal & hi_val,
                                               mreal delta = 0.)
        {
            mreal v1 = y1 - x1;
            mreal v2 = y2 - x2;
            mreal v3 = y3 - x3;
            
            mreal rCosPhi = v1 * n1 + v2 * n2 + v3 * n3;
            mreal rCosPsi = v1 * m1 + v2 * m2 + v3 * m3;
            mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 + delta;
            mreal r4 = r2 * r2;
            mreal r6 = r4 * r2;
            mreal mollifier = (1. - delta) * (1.- r2 * squared_inv_cut_off);
            mollifier = mollifier * mollifier;
            // Nasty trick to enforce vectorization without resorting to mypow or pos. Works only if intrinsic_dim is one of 1 or 2.
            mreal mul = t1 * r4 + t2 * r6;
            // The following line makes up approx 2/3 of this function's runtime! This is why we avoid pow as much as possible and replace it with mypow.
            mreal hi = mypow(r2, hi_exponent); // I got it down to this single call to pow. We might want to generate a lookup table for it...
            
            hi_val = mollifier * hi;
            
            fr_val = mollifier / (hi * mul);
            
            lo_val = 0.5 * mollifier * (rCosPhi * rCosPhi + rCosPsi * rCosPsi) / r4 * hi;
        }

        #pragma omp declare simd
        inline void ComputeInteraction( mreal x1, mreal x2, mreal x3, mreal p11, mreal p12, mreal p13, mreal p22, mreal p23, mreal p33,
                                        mreal y1, mreal y2, mreal y3, mreal q11, mreal q12, mreal q13, mreal q22, mreal q23, mreal q33,
                                        mreal t1, mreal t2, mreal hi_exponent,
                                        mreal & fr_val, mreal & lo_val, mreal & hi_val,
                                        mreal delta = 0.)
        {
            mreal v1 = y1 - x1;
            mreal v2 = y2 - x2;
            mreal v3 = y3 - x3;
            
            mreal rCosPhi2 = v1*(p11*v1 + p12*v2 + p13*v3) + v2*(p12*v1 + p22*v2 + p23*v3) + v3*(p13*v1 + p23*v2 + p33*v3);
            mreal rCosPsi2 = v1*(q11*v1 + q12*v2 + q13*v3) + v2*(q12*v1 + q22*v2 + q23*v3) + v3*(q13*v1 + q23*v2 + q33*v3);
            mreal r2 = v1 * v1 + v2 * v2 + v3 * v3 + delta;
            mreal r4 = r2 * r2;
            mreal r6 = r4 * r2;
            mreal mollifier = (1. - delta) * (1.- r2 * squared_inv_cut_off);
            mollifier = mollifier * mollifier;
            // Nasty trick to enforce vectorization without resorting to mypow or pos. Works only if intrinsic_dim is one of 1 or 2.
            mreal mul = t1 * r4 + t2 * r6;
            // The following line makes up approx 2/3 of this function's runtime! This is why we avoid pow as much as possible and replace it with mypow.
            mreal hi = mypow(r2, hi_exponent); // I got it down to this single call to pow. We might want to generate a lookup table for it...
            
            hi_val = mollifier * hi;
        
            fr_val = mollifier / (hi * mul);
            
            lo_val = 0.5 * mollifier * (rCosPhi2 + rCosPsi2) / r4 * hi;
        }
        
        
    }; //CutOffBlockClusterTree

    typedef std::shared_ptr<CutOffBlockClusterTree> CutOffBCTPtr;
} // namespace rsurfaces
