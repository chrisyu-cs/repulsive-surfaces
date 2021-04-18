#include "interaction_data.h"

namespace rsurfaces
{
    
    // General initialization; delays the computation of the sparse block matrix pattern to a later point.
    InteractionData::InteractionData( A_Vector<A_Deque<mint>> & idx, A_Vector<A_Deque<mint>> & jdx, const mint m_, const mint n_, bool upper_triangular_ )
    {
        ptic("InteractionData::InteractionData");
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
        
        A_Vector<mint> counters;
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                #pragma omp task
                {
                    safe_alloc( b_outer, 1 + b_m );
                    b_outer[0] = 0;
                }
                #pragma omp task
                {
                    safe_alloc( b_inner, b_nnz );
                }
                #pragma omp task
                {
                    counters = A_Vector<mint>(thread_count * b_m);
                }
                #pragma omp taskwait
            }
        }
        
        // using parallel count sort to sort the cluster (i,j)-pairs according to i.
        // storing counters of each i-index in thread-interleaved format
        // TODO: Improve data layout (transpose counts).
        #pragma omp parallel for num_threads( thread_count ) default( none ) shared( idx, counters, thread_count ) schedule( static, 1 )
        for( mint thread = 0; thread < thread_count; ++thread )
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
        
        #pragma omp parallel for num_threads( thread_count ) default(none) shared( b_m, b_outer, counters, thread_count) RAGGED_SCHEDULE
        for( mint i = 1; i < b_m + 1; ++i )
        {
            b_outer[i] = counters[ (thread_count * i) - 1 ];
        }
        // writing the j-indices into sep_column_indices
        // the counters array tells each thread where to write
        // since we have to decrement entries of counters array, we we have to loop in reverse order to make the sort stable in the j-indices.

        #pragma omp parallel for num_threads( thread_count ) default(none) shared( counters, idx, jdx, b_inner, thread_count) schedule( static, 1 )
        for( mint thread = 0; thread < thread_count; thread++)
        {
            for( mint k = idx[thread].size() - 1; k > - 1; --k )
            {
                mint pos = --counters[ thread_count * idx[thread][k] + thread  ];
                b_inner[pos] = jdx[thread][k];
            }
        }

        // We have to sort b_inner to be compatible with the CSR format.
        #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
        for( mint i = 0; i < b_m; ++i )
        {
            if(  b_outer[i+1] > b_outer[i] )
            {
                std::sort( b_inner + b_outer[i], b_inner + b_outer[i+1] );
            }
        }
        ptoc("InteractionData::InteractionData");
    }; // InteractionData Constructor


    // Allocate nonzero values for CSR matrices
    void InteractionData::Prepare_CSR()
    {
        ptic("InteractionData::Prepare_CSR");
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                #pragma omp task
                {
                    safe_alloc( hi_values, nnz );
                }
                #pragma omp task
                {
                    safe_alloc( lo_values, nnz );
                }
                #pragma omp task
                {
                    safe_alloc( fr_values, nnz );
                }
                #pragma omp taskwait
            }
        }
        
        // distribute workload
        BalanceWorkLoad( b_m, b_outer, thread_count, job_ptr);
        
        ptoc("InteractionData::Prepare_CSR");
    }


    // Allocate nonzero values _and_ compute block data for blocked matrices in CSR matrices
    void InteractionData::Prepare_CSR( mint b_m_, mint * b_row_ptr_, mint b_n_, mint * b_col_ptr_ )
    {
        ptic("InteractionData::Prepare_CSR( mint b_m_, mint * b_row_ptr_, mint b_n_, mint * b_col_ptr_ )");
        b_m = b_m_;
        b_n = b_n_;
        
        m = b_row_ptr_[b_m];
        n = b_col_ptr_[b_n];

        safe_alloc( outer, m + 1 );
        outer[0] = 0;
        safe_alloc( b_row_counters, b_m );
        
        // TODO: b_row_counters is needed only for computing block_ptr, which is only required for VBSR format (which we do not implement here).
        // TODO: Anyways, I leave it as uncommented code for potential later use.
        
        #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
        for( mint b_i = 0; b_i < b_m; ++b_i )
        {
            mint b_row_counter = 0;
            // Counting the number of entries per row in each block row to prepare outer.
            for( mint k =  b_outer[b_i], last = b_outer[b_i+1]; k < last; ++k )
            {
                mint b_j =  b_inner[k];
                mint n_b_j = b_col_ptr_[b_j + 1] - b_col_ptr_[b_j];
    //            b_row_counters[b_i] += n_b_j;
                b_row_counter += n_b_j;
            }
            b_row_counters[b_i] = b_row_counter;
            // Each row in a block row has the same number of entries.
            #pragma omp simd aligned( outer : ALIGN )
            for( mint k = b_row_ptr_[b_i]; k < b_row_ptr_[b_i+1]; ++k )
            {
                outer[k+1] = b_row_counter;
            }
        }

        partial_sum( outer, outer + m + 1);
        
        nnz = outer[m];
        
//        tic("Allocate nonzero values");
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                #pragma omp task
                {
                    safe_alloc( b_row_ptr, b_m + 1 );
                    #pragma omp simd
                    for( mint i = 0; i < b_m + 1; ++i )
                    {
                        b_row_ptr[i] = b_row_ptr_[i];
                    }
                }
                #pragma omp task
                {
                    safe_alloc( b_col_ptr, b_n + 1 );
                    #pragma omp simd
                    for( mint i = 0; i < b_n + 1; ++i )
                    {
                        b_col_ptr[i] = b_col_ptr_[i];
                    }
                }
                #pragma omp task
                {
                    safe_alloc( inner, nnz );
                }
                #pragma omp task
                {
                    safe_alloc( hi_values, nnz );
                }
                #pragma omp task
                {
                    safe_alloc( lo_values, nnz );
                }
                #pragma omp task
                {
                    safe_alloc( fr_values,nnz );
                }
                #pragma omp taskwait
            }
        }
//        toc("Allocate nonzero values");
        
        // Computing inner for CSR format.
        #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
        for( mint b_i = 0; b_i < b_m; ++ b_i )                  // we are going to loop over all rows in block fashion
        {
            // for each row i, write the column indices consecutively
            
            mint k_begin = b_outer[b_i];
            mint k_end   = b_outer[b_i+1];
            
            mint i_begin = b_row_ptr[b_i];
            mint i_end   = b_row_ptr[b_i+1];
            
            for( mint i = i_begin; i < i_end; ++i )                             // looping over all rows i  in block row b_i
            {
                mint ptr = outer[i];                            // get first nonzero position in row i; ptr will be used to keep track of the current position within inner
                
                for( mint k = k_begin; k < k_end; ++k )         // loop over all block in block row b_i
                {
                    mint b_j = b_inner[k];                      // we are in block {b_i, b_j}
                    
                    mint j_begin = b_col_ptr[b_j];
                    mint j_end   = b_col_ptr[b_j+1];
                    #pragma omp simd aligned( inner : ALIGN )
                    for( mint j = j_begin; j < j_end; ++j )     // write the column indices for row i
                    {
                        inner[ptr] = j;
                        ptr++;
                    }
                }
            }
        }

        safe_alloc( block_ptr, b_nnz );
        block_ptr[0] = 0;
        
        auto entries_before_block_row = A_Vector<mint>( b_m + 1 );

        // Creation of block pointers. Can be used for VBSR. Included here for debugging and for performance tests.
        
        #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
        for( mint b_i = 0; b_i < b_m; ++b_i )
        {
            //        mint entries_before = b_row_counters[ b_i ];
            mint mi = b_row_ptr[ b_i + 1 ] -  b_row_ptr[ b_i ];
            entries_before_block_row[b_i+1] = mi * b_row_counters[ b_i ];
        }
        
//        for( mint b_i = 0; b_i < b_m; ++b_i )
//        {
//            entries_before_block_row[b_i+1] += entries_before_block_row[b_i];
//        }
        
        partial_sum( &entries_before_block_row[0], &entries_before_block_row[0] +b_m + 1);
        
        #pragma omp parallel for num_threads(thread_count)
        for( mint b_i = 0; b_i < b_m; ++b_i )
        {
            mint entries_before_kth_block = entries_before_block_row[b_i];
            
            mint mi = b_row_ptr[ b_i + 1 ] -  b_row_ptr[ b_i ];
            
            for( mint k =  b_outer[ b_i ]; k < b_outer[ b_i + 1 ]; ++k)
            {
                // k-th block is {b_i, b_j}
                mint j =  b_inner[k];
                mint nj = b_col_ptr[j + 1] - b_col_ptr[j];
                // k-th block has size mi * nj
                entries_before_kth_block += mi * nj;
                block_ptr[k+1] = entries_before_kth_block;
            }
        }
        
        // distribute workload
        mint * b_row_acc_costs = nullptr;
        safe_alloc( b_row_acc_costs, b_m);
        b_row_acc_costs[0] = 0;
        #pragma omp parallel for simd num_threads(thread_count) aligned( b_row_acc_costs, b_row_counters, b_row_ptr : ALIGN )
        for( mint b_i = 0; b_i < b_m; ++b_i)
        {
            b_row_acc_costs[b_i+1] = b_row_counters[b_i] * (b_row_ptr[b_i + 1] - b_row_ptr[b_i]);
        }
        partial_sum( b_row_acc_costs, b_row_acc_costs + b_m + 1 );
        
        BalanceWorkLoad( b_m, b_row_acc_costs, thread_count, job_ptr);
        
        ptoc("InteractionData::Prepare_CSR( mint b_m_, mint * b_row_ptr_, mint b_n_, mint * b_col_ptr_ )");
    }
    
    // Allocate nonzero values _and_ compute block data for blocked matrices in VBSR matrices
    void InteractionData::Prepare_VBSR( mint b_m_, mint * b_row_ptr_, mint b_n_, mint * b_col_ptr_ )
    {
        ptic("InteractionData::Prepare_VBSR( mint b_m_, mint * b_row_ptr_, mint b_n_, mint * b_col_ptr_ )");
        b_m = b_m_;
        b_n = b_n_;
        
        m = b_row_ptr_[b_m];
        n = b_col_ptr_[b_n];

        safe_alloc( b_row_counters, b_m );
        
        // TODO: b_row_counters is needed only for computing block_ptr, which is only required for VBSR format (which we do not implement here).
        // TODO: Anyways, I leave it as uncommented code for potential later use.
        
        #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
        for( mint b_i = 0; b_i < b_m; ++b_i )
        {
            mint b_row_counter = 0;
            // Counting the number of entries per row in each block row to prepare outer.
            for( mint k =  b_outer[b_i], last = b_outer[b_i+1]; k < last; ++k )
            {
                mint b_j =  b_inner[k];
                mint n_b_j = b_col_ptr_[b_j + 1] - b_col_ptr_[b_j];
                b_row_counters[b_i] += n_b_j;
                b_row_counter += n_b_j;
            }
            b_row_counters[b_i] = b_row_counter;
        }
        
        //        tic("Allocate nonzero values");
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                #pragma omp task
                {
                    safe_alloc( b_row_ptr, b_m + 1 );
                    #pragma omp simd
                    for( mint i = 0; i < b_m + 1; ++i )
                    {
                        b_row_ptr[i] = b_row_ptr_[i];
                    }
                }
                #pragma omp task
                {
                    safe_alloc( b_col_ptr, b_n + 1 );
                    #pragma omp simd
                    for( mint i = 0; i < b_n + 1; ++i )
                    {
                        b_col_ptr[i] = b_col_ptr_[i];
                    }
                }
                #pragma omp task
                {
                    safe_alloc ( inner, nnz );
                }
                #pragma omp task
                {
                    safe_alloc( hi_values, nnz );
                }
                #pragma omp task
                {
                    safe_alloc( lo_values, nnz );
                }
                #pragma omp task
                {
                    safe_alloc( fr_values,nnz );
                }
                #pragma omp taskwait
            }
        }
        //        toc("Allocate nonzero values");
        
        // Computing inner for CSR format.
        #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
        for( mint b_i = 0; b_i < b_m; ++ b_i )                  // we are going to loop over all rows in block fashion
        {
            // for each row i, write the column indices consecutively
            
            mint k_begin = b_outer[b_i];
            mint k_end   = b_outer[b_i+1];
            
            mint i_begin = b_row_ptr[b_i];
            mint i_end   = b_row_ptr[b_i+1];
            
            for( mint i = i_begin; i < i_end; ++i )                             // looping over all rows i  in block row b_i
            {
                mint ptr = outer[i];                            // get first nonzero position in row i; ptr will be used to keep track of the current position within inner
                
                for( mint k = k_begin; k < k_end; ++k )         // loop over all block in block row b_i
                {
                    mint b_j = b_inner[k];                      // we are in block {b_i, b_j}
                    
                    mint j_begin = b_col_ptr[b_j];
                    mint j_end   = b_col_ptr[b_j+1];
                    #pragma omp simd aligned( inner : ALIGN )
                    for( mint j = j_begin; j < j_end; ++j )     // write the column indices for row i
                    {
                        inner[ptr] = j;
                        ptr++;
                    }
                }
            }
        }
        
        safe_alloc( block_ptr, b_nnz );
        block_ptr[0] = 0;
        
        auto entries_before_block_row = A_Vector<mint>( b_m + 1 );
        
        // Creation of block pointers. Can be used for VBSR. Included here for debugging and for performance tests.
        
        #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
        for( mint b_i = 0; b_i < b_m; ++b_i )
        {
            //        mint entries_before = b_row_counters[ b_i ];
            mint mi = b_row_ptr[ b_i + 1 ] -  b_row_ptr[ b_i ];
            entries_before_block_row[b_i+1] = mi * b_row_counters[ b_i ];
        }
        
        partial_sum( &entries_before_block_row[0], &entries_before_block_row[0] +b_m + 1);
        
        #pragma omp parallel for num_threads(thread_count) RAGGED_SCHEDULE
        for( mint b_i = 0; b_i < b_m; ++b_i )
        {
            mint entries_before_kth_block = entries_before_block_row[b_i];
            
            mint mi = b_row_ptr[ b_i + 1 ] -  b_row_ptr[ b_i ];
            
            for( mint k =  b_outer[ b_i ]; k < b_outer[ b_i + 1 ]; ++k)
            {
                // k-th block is {b_i, b_j}
                mint j =  b_inner[k];
                mint nj = b_col_ptr[j + 1] - b_col_ptr[j];
                // k-th block has size mi * nj
                entries_before_kth_block += mi * nj;
                block_ptr[k+1] = entries_before_kth_block;
            }
        }
        
        // distribute workload
        mint * b_row_acc_costs = nullptr;
        safe_alloc( b_row_acc_costs, b_m);
        b_row_acc_costs[0] = 0;
        #pragma omp parallel for simd num_threads(thread_count) aligned( b_row_acc_costs, b_row_counters, b_row_ptr : ALIGN )
        for( mint b_i = 0; b_i < b_m; ++b_i)
        {
            b_row_acc_costs[b_i+1] = b_row_counters[b_i] * (b_row_ptr[b_i + 1] - b_row_ptr[b_i]);
        }
        partial_sum( b_row_acc_costs, b_row_acc_costs + b_m + 1 );
        
        BalanceWorkLoad( b_m, b_row_acc_costs, thread_count, job_ptr);
        
        safe_free(b_row_acc_costs);
        ptoc("InteractionData::Prepare_VBSR( mint b_m_, mint * b_row_ptr_, mint b_n_, mint * b_col_ptr_ )");
    }
    
    void InteractionData::ApplyKernel_CSR_MKL( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor ) // sparse matrix-vector multiplication using mkl_sparse_d_mm
    {
        if( nnz == b_nnz )
        {
            ptic("ApplyKernel_CSR_MKL - far field");
        }
        else
        {
            ptic("ApplyKernel_CSR_MKL - near field");
        }
        
        if( T_input && S_output && OuterPtrB()[m] > 0 && values )
        {
            // Creation of handle for a sparse matrix in CSR format. This has almost no overhead. (Should be similar to Eigen's Map.)
            
            sparse_matrix_t A = NULL;
            sparse_status_t stat = mkl_sparse_d_create_csr ( &A, SPARSE_INDEX_BASE_ZERO, m, n, OuterPtrB(), OuterPtrE(), InnerPtr(), values );
            if (stat)
            {
                eprint("mkl_sparse_d_create_csr returned stat = " + std::to_string(stat) );
            }
            
            if( cols > 1 )
            {
//                tic("MKL sparse matrix-matrix multiplication: cols = " + std::to_string(cols) );
                stat = mkl_sparse_d_mm ( SPARSE_OPERATION_NON_TRANSPOSE, factor, A, descr, SPARSE_LAYOUT_ROW_MAJOR, T_input, cols, cols, 0., S_output, cols );
                if (stat)
                {
                    eprint("mkl_sparse_d_mm returned stat = " + std::to_string(stat) );
                }
//                toc("MKL sparse matrix-matrix multiplication: cols = " + std::to_string(cols) );
            }
            else
            {
//                tic("MKL sparse matrix-vector multiplication");
                stat = mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, factor, A, descr, T_input, 0, S_output);
                if (stat)
                {
                    eprint("mkl_sparse_d_mv returned stat = " + std::to_string(stat) );
                }
//                toc("MKL sparse matrix-vector multiplication");
            }

            stat = mkl_sparse_destroy(A);
            if (stat)
            {
                eprint("mkl_sparse_destroy returned stat = " + std::to_string(stat) );
            }
        }
        else
        {
            if( !T_input)
            {
                eprint("InteractionData::ApplyKernel_CSR_MKL: Input pointer is NULL. Doing nothing.");
            }
            if( !S_output)
            {
                eprint("InteractionData::ApplyKernel_CSR_MKL: Output pointer is NULL. Doing nothing.");
            }
            // !values or OuterPtrB()[m] == 0 are allowed to happen if there is no near field.
            if( !values && OuterPtrB()[m] == 0 )
            {
                #pragma omp parallel for simd aligned( S_output : ALIGN )
                for( mint i = 0; i < m * cols; ++i )
                {
                    S_output[i] = 0.;
                }
            }
        }
        if( nnz == b_nnz )
        {
            ptoc("ApplyKernel_CSR_MKL - far field");
        }
        else
        {
            ptoc("ApplyKernel_CSR_MKL - near field");
        }
    }; // ApplyKernel_CSR_MKL

    void InteractionData::ApplyKernel_CSR_Eigen( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor ) // sparse matrix-vector multiplication using Eigen
    {
        if( nnz == b_nnz )
        {
            ptic("ApplyKernel_CSR_Eigen - far field");
        }
        else
        {
            ptic("ApplyKernel_CSR_Eigen - near field");
        }
        Eigen::Map<EigenMatrixCSR> A ( m, n, nnz, OuterPtrB(), InnerPtr(), values );
        Eigen::Map<EigenMatrixRM>  B ( &T_input[0],  n, cols );
        Eigen::Map<EigenMatrixRM>  C ( &S_output[0], m, cols );

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
        
        if( nnz == b_nnz )
        {
            ptoc("ApplyKernel_CSR_Eigen - far field");
        }
        else
        {
            ptoc("ApplyKernel_CSR_Eigen - near field");
        }
    
    }; // ApplyKernel_CSR_Eigen

    
    void InteractionData::ApplyKernel_VBSR( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor ) // sparse matrix-vector multiplication using mkl_sparse_d_mm
    {

//
//        std::cout << "naive workloads = ";
//        for( mint k = 0; k < thread_count; ++k)
//        {
//            std::cout << b_row_acc_costs[naive_job_ptr[k+1]] -  b_row_acc_costs[naive_job_ptr[k]] << ", ";
//        }
//        std::cout << "\n" << std::endl;

        
        ptic("ApplyKernel_VBSR");
        #pragma omp parallel num_threads(thread_count)
        {
            mint b_i_begin = job_ptr[ omp_get_thread_num()     ];
            mint b_i_end   = job_ptr[ omp_get_thread_num() + 1 ];
            
            for( mint b_i = b_i_begin; b_i < b_i_end; ++ b_i)
            {
                // i-th chunk ui of output is a mi x cols matrix
                mint mi = b_row_ptr[b_i + 1] - b_row_ptr[b_i];
                mreal * ui = S_output + cols * b_row_ptr[b_i];
                
//                mint i_begin = b_row_ptr[ b_i     ];
//                mint i_end   = b_row_ptr[ b_i + 1 ];
                
                mint k_begin = b_outer[ b_i     ];
                mint k_end   = b_outer[ b_i + 1 ];
                
                for( mint k = k_begin; k < k_end; ++ k)
                {
                    mint b_j = b_inner[k];
                    mint j_begin = b_col_ptr[b_j];
                    mint j_end = b_col_ptr[b_j+1];
                    
                    // block Aij is a densemi x nj matrix
                    

                    mreal * Aij = values + block_ptr[k];
                    
                    mint nj = b_col_ptr[b_j + 1] - b_col_ptr[b_j];
                    
                    // j-th chunk vj of input is a nj x cols matrix
                    mreal * vj = T_input + cols * b_col_ptr[b_j];
                    cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, mi, cols, nj, 1., Aij, nj, vj, cols, 1., ui, cols );
//                                                                                                               ^
//                                                                                                               |
//                                                                                                               +--- add- into
                }
            }
        }
        ptoc("ApplyKernel_VBSR");
    }; // ApplyKernel_VBSR
    
    
    void InteractionData::ApplyKernel_Hybrid( mreal * values, mreal * T_input, mreal * S_output, mint cols, mreal factor ) // sparse matrix-vector multiplication using mkl_sparse_d_mm
    {

        mint max_len = *std::max_element(b_row_counters, b_row_counters + b_m);
        auto thread_input_buffers = A_Vector<A_Vector<mreal>> (thread_count);

        #pragma omp parallel num_threads(thread_count)
        {
            mint thread = omp_get_thread_num();
            thread_input_buffers[thread].resize( max_len * cols);
        }
        
        
        ptic("ApplyKernel_Hybrid");
        #pragma omp parallel num_threads(thread_count)
        {
            mint thread = omp_get_thread_num();
            mreal * v = &thread_input_buffers[thread][0];
            
            mint b_i_begin = job_ptr[ thread ];
            mint b_i_end   = job_ptr[ thread + 1 ];
            
            for( mint b_i = b_i_begin; b_i < b_i_end; ++ b_i)
            {
                mint i_begin = b_row_ptr[ b_i ];
                mint i_end   = b_row_ptr[ b_i + 1 ];
                

                mint k_begin = b_outer[ b_i     ];
                mint k_end   = b_outer[ b_i + 1 ];
                
                mreal * ptr = v;
                
                for( mint k = k_begin; k < k_end; ++ k)
                {
                    mint b_j = b_inner[k];
                    mint j_begin = b_col_ptr[b_j];
                    mint j_end = b_col_ptr[b_j+1];
            
                    mint nj = j_end - j_begin;

                    mreal * vj_begin = T_input + cols * j_begin;
                    mreal * vj_end   = T_input + cols * j_end;

                    std::copy( vj_begin, vj_end, ptr );
                    
//                    cblas_dcopy( cols * nj, T_input + cols * j_begin, 1, ptr, 1);
                    
                    ptr += cols * nj;
                }
                
                mint mi = i_end - i_begin;
                mreal * ui = S_output + cols * i_begin;
                mint row_nnz = b_row_counters[b_i];
                cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, mi, cols, row_nnz, 1., values + outer[i_begin], row_nnz, v, cols, 1., ui, cols );

            }
        }
        ptoc("ApplyKernel_Hybrid");
    }; // ApplyKernel_Hybrid
    
} // namespace rsurfaces



    
