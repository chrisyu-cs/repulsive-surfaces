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
    mint * restrict b_outer = NULL;             // block row pointers
    mint * restrict b_inner = NULL;             // block column indices
    
    // matrix sparsity pattern in CSR format -- to be used for CSR-gemm/CSR-symm implemented in mkl_sparse_d_mm
    mint m = 0;                                 // number of rows of the matrix
    mint n = 0;                                 // number of columns of the matrix
    mint nnz = 0;                               // number of nonzeros
    mint * restrict outer = NULL;               // row pointers
    mint * restrict inner = NULL;               // column indices
    
    // nonzero values
    mreal * restrict hi_values = NULL;          // nonzero values of high order kernel
    mreal * restrict lo_values = NULL;          // nonzero values of low order kernel
    mreal * restrict fr_values = NULL;          // nonzero values of fractional kernel in preconditioner
    
    matrix_descr descr;                     // sparse matrix descriptor for MKL's matrix-matrix routine ( mkl_sparse_d_mm )

    // Data for block matrics of variable block size. Used for the creation of the near field matrix
    
    mint * restrict b_row_ptr = NULL;           // accumulated block row sizes; used to compute position of output block; size = # rows +1;
    mint * restrict b_col_ptr = NULL;           // accumulated block colum sizes; used to compute position of input block; size = # colums +1;
    mint * restrict b_row_counters = NULL;      // b_row_counters[b_i] for block row b_i is the number of nonzero elements (which is constant among the rows contained in the block row_.
    mint * restrict block_ptr = NULL;           // block_ptr[k] is the index of the first nonzero entry of the k-th block
    
    void PrepareCSR();                                                                // Allocates nonzero values for matrix in CSR format.
    void PrepareCSR( mint b_m_, mint * b_row_ptr_, mint b_n_, mint * b_col_ptr_ );      // Allocates nonzero values  for blocked matrix (typically near field).

    inline void ApplyKernel( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor = 1. )
    {
        ApplyKernel_CSR_MKL( values, T_input, S_output, cols, factor );
//        ApplyKernel_CSR_Eigen( values, T_input, S_output, cols, factor );
    }; // ApplyKernel

    void ApplyKernel_CSR_MKL  ( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor = 1. );
    void ApplyKernel_CSR_Eigen( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor = 1. );
    
    mint * OuterPtrB() { if( nnz == b_nnz ){ return b_outer + 0; } else { return outer + 0; } };
    mint * OuterPtrE() { if( nnz == b_nnz ){ return b_outer + 1; } else { return outer + 1; } };
    mint * InnerPtr()  { if( nnz == b_nnz ){ return b_inner + 0; } else { return inner + 0; } };
    
    void sparse_d_mm_VBSR( const mreal * const restrict V, mreal * const restrict U, const mint cols );
    
    ~InteractionData(){
        mreal_free(hi_values);
        mreal_free(lo_values);
        mreal_free(fr_values);
        
        mint_free(outer);
        mint_free(inner);
        mint_free(b_outer);
        mint_free(b_inner);
        
        mint_free(b_row_ptr);
        mint_free(b_col_ptr);
        mint_free(b_row_counters);
        mint_free(block_ptr);
    };
}; //InteractionData


} // namespace rsurfaces
