#pragma once

#include "block_cluster_tree2_types.h"

namespace rsurfaces
{

struct InteractionData                      // Data structure for storing the sparse interaction matrices.
{                                           // To be owened only by instances of BlockClusterTree2. Dunno how to design it such that it cannot be seen from the outside.
public:
    InteractionData(){};

    // CSR initialization for ordinary sparse matrices.
    InteractionData( A_Vector<A_Deque<mint>> & idx, A_Vector<A_Deque<mint>> & jdx, const mint m_, const mint n_, bool upper_triangular_ );
    
    // CSR initialization for sparse block matrix.
    InteractionData( A_Vector<A_Deque<mint>> & idx, A_Vector<A_Deque<mint>> & jdx, const mint m_, const mint n_,
                    A_Vector<mint> & b_row_ptr_, A_Vector<mint> & b_col_ptr_, bool upper_triangular_ );
        
    ~InteractionData(){};

    mint thread_count = 1;
    bool upper_triangular = true;
    
    // block matrix sparsity pattern in CSR format -- to be used for interaction computations, visualization, debugging, and for VBSR-gemm/VBSR-symm (the latter is not implemented efficiently, yet)
    mint b_m = 0;                           // number of block rows of the matrix
    mint b_n = 0;                           // number of block columns of the matrix
    mint b_nnz = 0;                         // total number of blocks
    A_Vector<mint>  b_outer;                // block row pointers
    A_Vector<mint>  b_inner;                // block column indices
    
    // matrix sparsity pattern in CSR format -- to be used for CSR-gemm/CSR-symm implemented in mkl_sparse_d_mm
    mint m = 0;                             // number of rows of the matrix
    mint n = 0;                             // number of columns of the matrix
    mint nnz = 0;                           // number of nonzeros
    A_Vector<mint>  outer;                  // row pointers
    A_Vector<mint>  inner;                  // column indices
    
    // TODOL b_outer/outer and b_inner/b_outer are redundant for far field matrices; they are also redundant if
    
    // nonzero values
    A_Vector<mreal> hi_values;              // nonzero values of high order kernel
    A_Vector<mreal> lo_values;              // nonzero values of low order kernel
    A_Vector<mreal> fr_values;              // nonzero values of fractional kernel in preconditioner

    matrix_descr descr;                     // sparse matrix descriptor for MKL's matrix-matrix routine ( mkl_sparse_d_mm )

    // Data for block matrics of variable block size. Used for the creation of the near field matrix
    A_Vector<mint> b_row_ptr;               // accumulated block row sizes; used to compute position of output block; size = # rows +1;
    A_Vector<mint> b_col_ptr;               // accumulated block colum sizes; used to compute position of input block; size = # colums +1;
    A_Vector<mint> b_row_counters;          // b_row_counters[b_i] for block row b_i is the number of nonzero elements (which is constant among the rows contained in the block row_.
    A_Vector<mint> block_ptr;               // block_ptr[k] is the index of the first nonzero entry of the k-th block
    
    // mint Check() const;

    void ApplyKernel_CSR_MKL( A_Vector<mreal> & values, mreal const * const C_input, mreal * const C_output, const mint cols, const mreal factor = 1. );
    
    void ApplyKernel_CSR_Eigen( A_Vector<mreal> & values, mreal * C_input, mreal * C_output, const mint cols, const mreal factor = 1. );
    
}; //InteractionData

} // namespace rsurfaces
