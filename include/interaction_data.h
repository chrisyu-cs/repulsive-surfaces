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
    
    mint Check() const;

    void ApplyKernel_CSR_MKL( A_Vector<mreal> & values, mreal const * const C_input, mreal * const C_output, const mint cols, const mreal factor = 1. );
    
    void ApplyKernel_CSR_Eigen( A_Vector<mreal> & values, mreal * C_input, mreal * C_output, const mint cols, const mreal factor = 1. );
    
}; //InteractionData

// CRS initialization
InteractionData::InteractionData( A_Vector<A_Deque<mint>> & idx, A_Vector<A_Deque<mint>> & jdx, const mint m_, const mint n_, bool upper_triangular_ )
{
    thread_count = std::min( idx.size(), jdx.size());
    upper_triangular = upper_triangular_;
    b_m = m = m_;
    b_n = n = n_;
    
    nnz = 0;
    for( mint thread = 0; thread < thread_count; ++thread )
    {
        nnz += idx[thread].size();
    }
    b_nnz = nnz;
    
    outer =  A_Vector<mint>( 1 + m );
    inner =  A_Vector<mint>( nnz );
    
    b_outer =  A_Vector<mint>( 1 + b_m );
    b_inner =  A_Vector<mint>( b_nnz );
    
    hi_values   = A_Vector<mreal>( nnz );
    lo_values   = A_Vector<mreal>( nnz );
    fr_values = A_Vector<mreal>( nnz );
    
    if ( upper_triangular)
    {
        descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
        descr.mode = SPARSE_FILL_MODE_UPPER;
        descr.diag = SPARSE_DIAG_NON_UNIT;
    }
    else
    {
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        descr.diag = SPARSE_DIAG_NON_UNIT;
    }

    // using parallel count sort to sort the cluster (i,j)-pairs according to i.

    A_Vector<mint>  counters = A_Vector<mint>(thread_count * b_m);

    // storing counters of each i-index in thread-interleaved format
    // TODO: Improve data layout (transpose counts).
    #pragma omp parallel for num_threads( thread_count ) default( none ) shared( idx, counters )
    for(mint thread = 0; thread < thread_count; ++thread )
    {
        for( mint k = 0, last = idx[thread].size(); k < last; ++k )
        {
            ++counters[ thread_count * idx[thread][k] + thread ];
        }
    }
    

    for( mint i = 0, last = thread_count * b_m - 1; i < last; ++i)
    {
        counters[ i+1 ] += counters[i];
    }

    #pragma omp parallel for num_threads( thread_count ) default(none) shared( b_m, outer, b_outer, counters, thread_count)
    for( mint i = 1; i < b_m + 1; ++i )
    {
        mint temp = counters[ (thread_count * i) -1 ];
        outer[i] = temp;
        b_outer[i] = temp;
    }
    // writing the j-indices into sep_column_indices
    // the counters array tells each thread where to write
    // since we have to decrement entries of counters array, we we have to loop in reverse order to make the sort stable in the j-indices.

    #pragma omp parallel for num_threads( thread_count ) default(none) shared( counters, idx, jdx, inner, b_inner, thread_count)
    for( mint thread = 0; thread < thread_count; thread++)
    {
        for( mint k = idx[thread].size() - 1; k > -1; --k )
        {
            mint pos = --counters[ thread_count * idx[thread][k] + thread  ];
            mint temp =  jdx[thread][k];
            b_inner[pos] = temp;
            inner[pos] = temp;
        }
    }

    // We have to sort inner and b_inner to be compatible with the CSR format.

    #pragma omp parallel for
    for( mint i = 0; i < b_m; ++i )
    {
        std::sort( b_inner.begin() + b_outer[i], b_inner.begin() + b_outer[i+1] );
        std::copy( b_inner.begin() + b_outer[i], b_inner.begin() + b_outer[i+1], inner.begin() + b_outer[i] );
    }

}; // InteractionData Constructor

// CSR initialization for sparse block matrix
InteractionData::InteractionData(
                                 A_Vector<A_Deque<mint>> & idx,
                                 A_Vector<A_Deque<mint>> & jdx,
                                 const mint m_,
                                 const mint n_,
                                 A_Vector<mint> & b_row_ptr_,
                                 A_Vector<mint> & b_col_ptr_,
                                 bool upper_triangular_ )
{
    thread_count  = idx.size();
    upper_triangular = upper_triangular_;
    b_m = b_row_ptr_.size()-1;
    b_n = b_col_ptr_.size()-1;
    m = m_;
    n = n_;
    
    b_nnz = 0;
    for( mint thread = 0; thread < thread_count; ++thread )
    {
        b_nnz += idx[thread].size();
    }
    
    b_row_ptr = A_Vector<mint>( b_m + 1 );
    std::copy( b_row_ptr_.begin(), b_row_ptr_.end() , b_row_ptr.begin() );
    b_col_ptr = A_Vector<mint>( b_n + 1 );
    std::copy( b_col_ptr_.begin(), b_col_ptr_.end() , b_col_ptr.begin() );
    

    b_outer =  A_Vector<mint>( b_m + 1 );
    // TODO: This allocation can potentially take some time. Might be a good idea to create this as a task, concurrent with other tasks.
    b_inner =  A_Vector<mint>( b_nnz );

    if ( upper_triangular )
    {
        descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
        descr.mode = SPARSE_FILL_MODE_UPPER;
        descr.diag = SPARSE_DIAG_NON_UNIT;
    }
    else
    {
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        descr.diag = SPARSE_DIAG_NON_UNIT;
    }
    
    // using parallel count sort to sort the cluster (i,j)-pairs according to i.
    
    A_Vector<mint> counters = A_Vector<mint>( thread_count * b_m );

    // storing counters of each i-index in thread-interleaved format
    // TODO: Improve data layout (transpose counts).
    #pragma omp parallel for num_threads( thread_count ) default(none) shared(idx, counters, thread_count)
    for( mint thread = 0; thread < thread_count; thread++)
    {
        for( mint k = 0, last = idx[thread].size(); k < last; k++ )
        {
            ++counters[ thread_count * idx[thread][k] + thread ];
        }
    }
    
    for( mint i = 0, last = thread_count * b_m - 1; i < last ; ++i )
    {
        counters[ i+1 ] += counters[i];
    }

    #pragma omp parallel for num_threads( thread_count ) default(none) shared( b_m, outer, counters, thread_count)
    for( mint i = 1; i < b_m + 1; ++i )
    {
        b_outer[i] = counters[ (thread_count * i) -1 ];
    }
    // writing the j-indices into sep_column_indices
    // the counters array tells each thread where to write
    // since we have to decrement entries of counters array, we we have to loop in reverse order to make the sort stable in the j-indices.
    
    #pragma omp parallel for num_threads( thread_count ) default(none) shared( counters, idx, jdx, b_inner, thread_count)
    for( mint thread = 0; thread < thread_count; ++thread )
    {
        for( mint k = idx[thread].size() - 1; k > -1; --k )
        {
            mint pos = --counters[ thread_count * idx[thread][k] + thread  ];
            
            b_inner[pos] = jdx[thread][k];
        }
    }
    b_row_counters = A_Vector<mint> ( b_m ) ;
    outer =  A_Vector<mint>( m + 1 );

    #pragma omp parallel for
    for( mint i = 0; i < b_m; ++i )
    {
        // We have to sort b_inner to be compatible with the CSR format.
        std::sort( b_inner.begin() + b_outer[i], b_inner.begin() + b_outer[i+1] );
        
        // Counting the number of entries per row in each block row to prepare outer.
        for( mint k =  b_outer[i]; k < b_outer[i+1]; ++k )
        {
            mint j =  b_inner[k];
            mint nj = b_col_ptr[j + 1] - b_col_ptr[j];
            b_row_counters[i] += nj;
        }
        // Each row in a block row has the same number of entries.
        for( mint k = b_row_ptr[i]; k < b_row_ptr[i+1]; ++k )
        {
            outer[k+1] = b_row_counters[i];
        }
    }
    // Now outer[k+1] contains the number of entries in k-th row. Accumulating to get the true row pointers.
    for( mint k = 1, last = m + 1; k < last ; ++k )
    {
        outer[k] += outer[k-1];
    }

    nnz = outer.back();                                                     //Outer cannot be empty.
    
    tic("Allocate inner and nonzero values");
    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            #pragma omp task
            {
                inner     = A_Vector<mint>( nnz );
            }
            #pragma omp task
            {
                hi_values = A_Vector<mreal>( nnz );
            }
            #pragma omp task
            {
                lo_values = A_Vector<mreal>( nnz );
            }
            #pragma omp task
            {
                fr_values = A_Vector<mreal>( nnz );
            }
            #pragma omp taskwait
        }
    }
    toc("Allocate inner and nonzero values");
    
    block_ptr = A_Vector<mint>( b_nnz );
          
    tic("Computing inner");
    // Computing inner for CSR format.
    #pragma omp parallel for
    for( mint b_i = 0; b_i <  b_m; ++ b_i )                                 // we are going to loop over all rows in block fashion
    {
        // for each row i, write the column indices consecutively
        
        mint k_begin = b_outer[b_i];
        mint k_end   = b_outer[b_i+1];
        
        for( mint i = b_row_ptr[b_i]; i < b_row_ptr[b_i+1]; ++i )           // looping over all rows i  in block row b_i
        {
            mint ptr = outer[i];                                            // get first nonzero position in row i; ptr will be used to keep track of the current position within inner
            
            for( mint k =  k_begin; k < k_end; ++k )                        // loop over all block in block row b_i
            {
                mint b_j =  b_inner[k];                                     // we are in block {b_i, b_j}
                for( mint j = b_col_ptr[b_j]; j < b_col_ptr[b_j+1]; ++j )   // write the column indices for row i
                {
                    inner[ptr] = j;
                    ptr++;
                }
            }
        }
    }
    toc("Computing inner");

    // Creation of block pointers. Can be used for VBSR. Included here for debugging and for performance tests.
    #pragma omp parallel for
    for( mint b_i = 0; b_i <  b_m; ++b_i )
    {
        mint entries_before = b_row_counters[ b_i ];
        mint mi = b_row_ptr[ b_i + 1 ] -  b_row_ptr[ b_i ];
        for( mint k =  b_outer[ b_i ]; k < b_outer[ b_i + 1 ]; ++k)
        {
            mint j =  b_inner[k];
            mint nj = b_col_ptr[j + 1] - b_col_ptr[j];
            entries_before += mi * nj;
            block_ptr[k+1] = entries_before;
        }
    }
}; // InteractionData Constructor

void InteractionData::ApplyKernel_CSR_MKL( A_Vector<mreal> & values, mreal const * const T_input, mreal * const S_output, const mint cols, const mreal factor ) // sparse matrix-vector multiplication using mkl_sparse_d_mm
{
    if( outer[m]>0 )
    {
        // Creation of handle for a sparse matrix in CSR format. This has almost no overhead. (Should be similar to Eigen's Map.)
        sparse_matrix_t A = NULL;
        sparse_status_t stat = mkl_sparse_d_create_csr ( &A, SPARSE_INDEX_BASE_ZERO, m, n, &outer[0], &outer[1], &inner[0], &values[0] );
        if (stat)
        {
            eprint("mkl_sparse_d_create_csr returned stat = " + std::to_string(stat) );
        }
        
        // sparse_matrix_t A -- MKL sparse matrix handle for matrix of size m x n
        // mreal * B -- pointer to  n x cols matrix, storing the data on the clusters before multiplication
        // mreal * C -- pointer to  m x cols matrix, storing the data on the clusters after multiplication.
        // performs C = alpha*op(A)*B + beta*C:
        // sparse_status_t mkl_sparse_d_mm (sparse_operation_t operation, mreal alpha, const sparse_matrix_t A, struct matrix_descr descr, sparse_layout_t layout, const mreal *B, mint columns, mint ldB, mreal beta, mreal *C, mint ldC);
        // columns = ldB = ldC = number of columns of B and C.
        
        if( cols > 1 )
        {
            //        tic("MKL sparse matrix-matrix multiplication: cols = " + std::to_string(cols) );
            stat = mkl_sparse_d_mm ( SPARSE_OPERATION_NON_TRANSPOSE, factor, A, descr, SPARSE_LAYOUT_ROW_MAJOR, &T_input[0], cols, cols, 0., &S_output[0], cols );
            if (stat)
            {
                eprint("mkl_sparse_d_mm returned stat = " + std::to_string(stat) );
            }
            //        toc("MKL sparse matrix-matrix multiplication: cols = " + std::to_string(cols) );
        }
        else
        {
            //        tic("MKL sparse matrix-vector multiplication");
            stat = mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, factor, A, descr, &T_input[0], 0, &S_output[0]);
            if (stat)
            {
                eprint("mkl_sparse_d_mv returned stat = " + std::to_string(stat) );
            }
            //        toc("MKL sparse matrix-vector multiplication");
        }
        
        stat = mkl_sparse_destroy(A);
        if (stat)
        {
            eprint("mkl_sparse_destroy returned stat = " + std::to_string(stat) );
        }
    }
    else
    {
        print("MKLSparseMatrix::Multiply: No nonzeroes found. Doing nothing.");
    }
}; // ApplyKernel_CSR_MKL

void InteractionData::ApplyKernel_CSR_Eigen( A_Vector<mreal> & values, mreal * T_input, mreal * S_output, const mint cols, const mreal factor ) // sparse matrix-vector multiplication using Eigen
{
    
//    wprint("InteractionData::ApplyKernel_CSR_Eigen is not thouroughly tested, yet.");
    Eigen::Map<EigenMatrixCSR> A ( m, n, nnz, &outer[0], &inner[0], &values[0] );
    Eigen::Map<EigenMatrixRM>  B ( &T_input[0],  n, cols );
    Eigen::Map<EigenMatrixRM>  C ( &S_output[0], m, cols );

//    tic("Eigen matrix-matrix multiplication: cols = " + std::to_string(cols) );
    if ( upper_triangular )
    {
        C = A.selfadjointView<Eigen::Upper>() * B;
    }
    else
    {
        C = A * B;
    }
    if (factor != 1.)
    {
        C *= factor;
    }
//    toc("Eigen matrix-matrix multiplication: cols = " + std::to_string(cols) );
}; // ApplyKernel_CSR_Eigen


//// Some boring matrix checker routine...
//mint InteractionData::Check() const
//{
//    mint err = 0;
//
//    print("{ b_m, b_n } = {" + std::to_string(b_m) + ", " + std::to_string(b_n) +" }");
//
//    if( b_outer.size() != b_m + 1 )
//    {
//        err = 1;
//        print("ERROR: b_outer.size() != b_m + 1");
//    }
//
//    if( b_outer[0] != 0 )
//    {
//        err = 1;
//        print("ERROR: b_outer[0] != 0");
//        print("ERROR: b_outer[0] = " + std::to_string(b_outer[0]) );
//    }
//
//    if( b_outer.back() != b_nnz )
//    {
//        err = 1;
//        print("ERROR: b_outer.back() = " + std::to_string(b_outer.back()) + " !=  " + std::to_string(b_outer.back()) + " = b_nnz");
//    }
//
//    if( b_inner.size() != b_nnz )
//    {
//        err = 1;
//        print("ERROR: b_inner.size() != b_nnz");
//    }
//
//    bool b;
//
//    b = true;
//    for( mint i = 0; i < b_outer.size() - 1; ++i )
//    {
//        b = b && ( b_outer[i] <= b_outer[i+1] );
//    }
//
//    if( !b )
//    {
//        err = 1;
//        print("ERROR: b_inner is not ascending");
//    }
//
//    b = true;
//    for( mint i = 0; i < b_outer.size() - 1; ++i )
//    {
//        for( mint k = b_outer[i]; k < b_outer[i+1] - 1; ++k )
//        {
//            b = b && ( b_inner[k] <= b_inner[k+1] );
//        }
//    }
//    if( !b )
//    {
//        err = 1;
//        print("ERROR: b_outer is not row-wise ascending");
//    }
//
//    b = true;
//    for( mint i = 0; i < b_inner.size(); ++i )
//    {
//        b = b && (0 <= b_inner[i]) && (b_inner[i] <= b_n);
//    }
//    if( !b )
//    {
//        err = 1;
//        print("ERROR: b_inner is out of bounds");
//    }
//
//
//
//
//    print("{ m, n } = {" + std::to_string(m) + ", " + std::to_string(n) +" }");
//    if (outer.size() != m + 1)
//    {
//        err = 1;
//        print("ERROR: outer.size() != m + 1");
//    }
//
//    if( outer[0] != 0 )
//    {
//        err = 1;
//        print("ERROR: outer[0] != 0");
//        print("ERROR: outer[0] = " + std::to_string(outer[0]) );
//    }
//
//
//
//    if( outer.back() != nnz )
//    {
//        err = 1;
//        print("ERROR: outer.back() = " + std::to_string(outer.back()) + " !=  " + std::to_string(nnz) + " = nnz");
//    }
//
//    if( inner.size() != nnz )
//    {
//        err = 1;
//        print("ERROR: inner.size() != nnz");
//    }
//
//
//    b = true;
//    for( mint i = 0; i < outer.size() - 1; ++i )
//    {
//        b = b && ( outer[i] <= outer[i+1] );
//    }
//
//    if( !b )
//    {
//        err = 1;
//        print("ERROR: inner is not ascending");
//    }
//
//    b = true;
//    for( mint i = 0; i < outer.size() - 1; ++i )
//    {
//        for( mint k = outer[i]; k < outer[i+1] - 1; ++k )
//        {
//            b = b && ( inner[k] <= inner[k+1] );
//        }
//    }
//    if( !b )
//    {
//        err = 1;
//        print("ERROR: outer is not row-wise ascending");
//    }
//
//    b = true;
//    for( mint i = 0; i < inner.size(); ++i )
//    {
//        b = b && (0 <= inner[i]) && (inner[i] <= n);
//    }
//    if( !b )
//    {
//        err = 1;
//        print("ERROR: inner is is out of bounds");
//    }
//
//    return err;
//}; // Check
//EigenMatrixCSR

} // namespace rsurfaces
