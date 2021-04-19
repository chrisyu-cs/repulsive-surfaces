#pragma once

#include "optimized_bct_types.h"

namespace rsurfaces
{

struct InteractionData                      // Data structure for storing the sparse interaction matrices.
{                                           // To be owned only by instances of BlockClusterTree. Dunno how to design it such that it cannot be seen from the outside.
public:
    InteractionData(){};

    // CSR initialization for ordinary sparse matrices.
    InteractionData( A_Vector<A_Deque<mint>> & idx, A_Vector<A_Deque<mint>> & jdx, const mint m_, const mint n_, bool upper_triangular_ );
    
//    // CSR initialization for sparse block matrix.
//    InteractionData( A_Vector<A_Deque<mint>> & idx, A_Vector<A_Deque<mint>> & jdx, const mint m_, const mint n_,
//                     A_Vector<mint> & b_row_ptr_, A_Vector<mint> & b_col_ptr_, bool upper_triangular_ );

    mint thread_count = 1;
    bool upper_triangular = true;
    
    // block matrix sparsity pattern in CSR format -- to be used for interaction computations, visualization, debugging, and for VBSR-gemm/VBSR-symm (the latter is not implemented efficiently, yet)
    mint b_m = 0;                               // number of block rows of the matrix
    mint b_n = 0;                               // number of block columns of the matrix
    mint b_nnz = 0;                             // total number of blocks
    mint * restrict b_outer = nullptr;             // block row pointers
    mint * restrict b_inner = nullptr;             // block column indices
    
    // matrix sparsity pattern in CSR format -- to be used for CSR-gemm/CSR-symm implemented in mkl_sparse_d_mm
    mint m = 0;                                 // number of rows of the matrix
    mint n = 0;                                 // number of columns of the matrix
    mint nnz = 0;                               // number of nonzeros
    mint * restrict outer = nullptr;               // row pointers
    mint * restrict inner = nullptr;               // column indices
    
    // Scaling parameters for the matrix-vector product.
    // E.g. one can perform  u = hi_factor * A_hi * v. With MKL, this comes at no cost, because it is fused into the matrix multiplications anyways.
    mreal hi_factor = 1.;
    mreal lo_factor = 1.;
    mreal fr_factor = 1.;
    
    // nonzero values
    mreal * restrict hi_values = nullptr;          // nonzero values of high order kernel
    mreal * restrict lo_values = nullptr;          // nonzero values of low order kernel
    mreal * restrict fr_values = nullptr;          // nonzero values of fractional kernel in preconditioner
    
    matrix_descr descr;                     // sparse matrix descriptor for MKL's matrix-matrix routine ( mkl_sparse_d_mm )

    // Data for block matrics of variable block size. Used for the creation of the near field matrix
    
    mint * restrict b_row_ptr = nullptr;           // accumulated block row sizes; used to compute position of output block; size = # rows +1;
    mint * restrict b_col_ptr = nullptr;           // accumulated block colum sizes; used to compute position of input block; size = # colums +1;
    mint * restrict b_row_counters = nullptr;      // b_row_counters[b_i] for block row b_i is the number of nonzero elements
                                                   // (which is constant among the rows contained in the block row_.
    mint * restrict block_ptr = nullptr;           // block_ptr[k] is the index of the first nonzero entry of the k-th block
    
    void Prepare_CSR();                                                                  // Allocates nonzero values for matrix in CSR format.
    void Prepare_CSR( mint b_m_, mint * b_row_ptr_, mint b_n_, mint * b_col_ptr_ );      // Allocates nonzero values  for blocked matrix (typically near field).
    void Prepare_VBSR( mint b_m_, mint * b_row_ptr_, mint b_n_, mint * b_col_ptr_ );     // Allocates nonzero values  for blocked matrix (typically near field).

    
    inline void ApplyKernel( BCTKernelType type, mreal * T_input, mreal * S_output, mint cols, mreal factor = 1., NearFieldMultiplicationAlgorithm mult_alg = NearFieldMultiplicationAlgorithm::MKL_CSR)
    {
        ptic("ApplyKernel");
        
        switch (type)
        {
            case BCTKernelType::FractionalOnly:
            {
                ApplyKernel( fr_values, T_input, S_output, cols, factor * fr_factor, mult_alg );
                break;
            }
            case BCTKernelType::HighOrder:
            {
                ApplyKernel( hi_values, T_input, S_output, cols, factor * hi_factor, mult_alg );
                break;
            }
            case BCTKernelType::LowOrder:
            {
                ApplyKernel( lo_values, T_input, S_output, cols, factor * lo_factor, mult_alg );
                break;
            }
            default:
            {
                eprint("ApplyKernel: Unknown kernel. Doing nothing.");
                break;
            }
        }
        
        ptoc("ApplyKernel");
    }; // ApplyKernel
    
    inline void ApplyKernel( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor = 1., NearFieldMultiplicationAlgorithm mult_alg = NearFieldMultiplicationAlgorithm::MKL_CSR )
    {

        if( factor != 0. )
        {
            if( nnz == b_nnz)
            {
                switch (mult_alg)
                {
                    case NearFieldMultiplicationAlgorithm::MKL_CSR :
//                        print("ApplyKernel_CSR_MKL - near field");
                        ApplyKernel_CSR_MKL( values, T_input, S_output, cols, factor );
                        break;
                    case NearFieldMultiplicationAlgorithm::Eigen :
//                        print("ApplyKernel_CSR_Eigen - near field");
                        ApplyKernel_CSR_Eigen( values, T_input, S_output, cols, factor );
                        break;
                    default:
//                        print("ApplyKernel_CSR_MKL - near field (default)");
                        ApplyKernel_CSR_MKL( values, T_input, S_output, cols, factor );
                        break;
                }
            }
            else
            {
                switch (mult_alg)
                {
                    case NearFieldMultiplicationAlgorithm::MKL_CSR :
//                        print("ApplyKernel_CSR_MKL - far field");
                        ApplyKernel_CSR_MKL( values, T_input, S_output, cols, factor );
                        break;
                    case NearFieldMultiplicationAlgorithm::Hybrid :
//                        print("ApplyKernel_CSR_Hybrid - far field");
                        ApplyKernel_Hybrid( values, T_input, S_output, cols, factor );
                        break;
                    case NearFieldMultiplicationAlgorithm::Eigen :
//                        print("ApplyKernel_CSR_Eigen - far field");
                        ApplyKernel_CSR_Eigen( values, T_input, S_output, cols, factor );
                        break;
                    default:
//                        print("ApplyKernel_CSR_MKL - far field (default)");
                        ApplyKernel_CSR_MKL( values, T_input, S_output, cols, factor );
                        break;
                }
            }
        }
        else
        {
            #pragma omp parallel for simd aligned( S_output : ALIGN)
            for( mint i = 0; i < m * cols; ++i )
            {
                S_output[i] = 0.;
            }
        }
    }; // ApplyKernel

    mint * job_ptr = nullptr;
    
    void ApplyKernel_VBSR     ( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor = 1. );
    void ApplyKernel_CSR_MKL  ( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor = 1. );
    void ApplyKernel_CSR_Eigen( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor = 1. );
    void ApplyKernel_Hybrid   ( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor = 1. ) ;
    
    mint * OuterPtrB() { if( nnz == b_nnz ){ return b_outer + 0; } else { return outer + 0; } };
    mint * OuterPtrE() { if( nnz == b_nnz ){ return b_outer + 1; } else { return outer + 1; } };
    mint * InnerPtr()  { if( nnz == b_nnz ){ return b_inner + 0; } else { return inner + 0; } };
    
//    void sparse_d_mm_VBSR( const mreal * const restrict V, mreal * const restrict U, const mint cols );
    
    ~InteractionData(){
        ptic("~InteractionData");
        
        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                {
                    safe_free(hi_values);
                }
                #pragma omp task
                {
                    safe_free(lo_values);
                }
                #pragma omp task
                {
                    safe_free(fr_values);
                }
                #pragma omp task
                {
                    safe_free(outer);
                }
                #pragma omp task
                {
                    safe_free(inner);
                }
                #pragma omp task
                {
                    safe_free(b_outer);
                }
                #pragma omp task
                {
                    safe_free(b_inner);
                }
                #pragma omp task
                {
                    safe_free(b_row_ptr);
                }
                #pragma omp task
                {
                    safe_free(b_col_ptr);
                }
                #pragma omp task
                {
                    safe_free(b_row_counters);
                }
                #pragma omp task
                {
                    safe_free(block_ptr);
                }
                #pragma omp task
                {
                    safe_free(job_ptr);
                }
                #pragma omp taskwait
            }
        }
        ptoc("~InteractionData");
    };
}; //InteractionData


} // namespace rsurfaces
