#pragma once

#include <mkl.h>
#include <tbb/cache_aligned_allocator.h>
#include <algorithm>
#include <omp.h>
#include <deque>
#include <vector>
#include <iterator>
#include <memory>
#include <unistd.h>
#include <string>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iostream>

#define restrict __restrict

namespace rsurfaces
{

// "In know only two types: integers and doubles..."
typedef MKL_INT mint;   // "machine integer" -- ensuring that we use the integer type requested by MKL. I find "MKL_INT" a bit clunky, though.
typedef double mreal;   // "machine real"

typedef Eigen::SparseMatrix<mreal, Eigen::RowMajor, mint> EigenMatrixCSR;
typedef Eigen::SparseMatrix<mreal> EigenMatrixCSC;

typedef Eigen::Matrix<mreal, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixRM;
typedef Eigen::MatrixXd EigenMatrixCM;

std::deque < std::chrono::time_point<std::chrono::steady_clock> > time_stack;

std::chrono::time_point<std::chrono::steady_clock> start_time, stop_time;

inline void print( std::string s ){
    std::cout << ( std::string(time_stack.size()+1, '\t') + s) << std::endl;
}

inline void eprint( std::string s ){
    std::cout << ( std::string(time_stack.size()+1, '\t') + "ERROR: " + s ) << std::endl;
}

inline void wprint( std::string s ){
    std::cout << ( std::string(time_stack.size()+1, '\t') + "WARNING: " + s ) << std::endl;
}


inline void tic( std::string s ){
    time_stack.push_back( std::chrono::steady_clock::now() );
    std::cout << ( std::string(time_stack.size(), '\t') + s + "..." ) << std::endl;
}

inline void toc( std::string s ){
    if( !time_stack.empty() ){
        auto start_time = time_stack.back();
        auto stop_time = std::chrono::steady_clock::now();
        std::cout << ( std::string(time_stack.size(), '\t') + std::to_string ( ( std::chrono::duration < double >  (stop_time - start_time).count() ) ) + " s." ) << std::endl;
        time_stack.pop_back();
    }
    else
    {
        std::cout << ("Unmatched toc detected. Label =  " + s) << std::endl;
    }
}


// I am not that knowledgable about allocators; tbb::cache_aligned_allocator seemed to work well for allocating thread-owned vectors. And I simply used it for the rest, too, because should also provide good alignment for SIMD instructions (used in MKL routines). DO NOT USE A_Vector OR A_Deque FOR MANY SMALL ARRAYS. I typically allicate only very large arrays, so the exrta memory consumption should not be an issue.
template <typename T>
using A_Vector = std::vector<T, tbb::cache_aligned_allocator<T>>;

template <typename T>
using A_Deque = std::deque<T, tbb::cache_aligned_allocator<T>>;

//template <typename T>
//using A_Vector = std::vector<T>;   // about 10% more performance with cache-alined storage
//
//template <typename T>
//using A_Deque = std::deque<T>;


struct MKLSparseMatrix //A container to hold generic sparse array data and to perform MKL matrix-matrix-multiplication routines
{
    mint m = 0;
    mint n = 0;
    mint nnz = 0;
    A_Vector<mint> outer;
    A_Vector<mint> inner;
    A_Vector<mreal> values;
    
    matrix_descr descr;
    
    //    MKLSparseMatrix( SparseMatrix A ) : MKLSparseMatrix( A.rows(), A.cols(), A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr() ){}; //dunno why this does not work...
    
    MKLSparseMatrix( const mint m_, const mint n_, const mint nnz_ )
    {
        m = m_;
        n = n_;
        nnz = nnz_;
        
        outer  = A_Vector<mint> ( m + 1 );
        inner  = A_Vector<mint> ( nnz );
        values = A_Vector<mreal>( nnz );
        
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        descr.diag = SPARSE_DIAG_NON_UNIT;
        
    };
    
    MKLSparseMatrix( const mint m_, const mint n_, mint * outer_, mint * inner_, mreal * values_ )
    {
        m = m_;
        n = n_;
        nnz = outer_[m];
        
        outer  = A_Vector<mint> ( outer_ , outer_  + m + 1 );
        inner  = A_Vector<mint> ( inner_ , inner_  + nnz   );
        values = A_Vector<mreal>( values_, values_ + nnz   );
        
        
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        descr.diag = SPARSE_DIAG_NON_UNIT;
        
        //        outer  = A_Vector<mint> ( m + 1 );
        //        inner  = A_Vector<mint> ( nnz );
        //        values = A_Vector<mreal>( nnz );
        //
        //        #pragma omp parallel for
        //        for( mint i = 0; i < m + 1; ++i )
        //        {
        //            outer[i]  = outer_ [i];
        //        }
        //        #pragma omp parallel for
        //        for( mint i = 0; i < nnz; ++i )
        //        {
        //            inner[i]  = inner_ [i];
        //            values[i] = values_[i];
        //        }
        
    };
    
    MKLSparseMatrix(){};
    
    ~MKLSparseMatrix(){};
    
    // TODO: Better error handling
    
    void Multiply( A_Vector<mreal> & input, A_Vector<mreal> & output, mint cols, bool addToResult = false )
    {
        Multiply( &input[0], &output[0], cols, addToResult );
    }
    
    void Multiply( mreal * input, mreal * output, mint cols, bool addToResult = false )
    {
        
        if( outer[m]>0 )
        {
            sparse_status_t stat;
            
            sparse_matrix_t A = NULL;
            
            stat = mkl_sparse_d_create_csr ( &A, SPARSE_INDEX_BASE_ZERO, m, n, &outer[0], &outer[1], &inner[0], &values[0] );
            if (stat)
            {
                eprint(" in MKLSparseMatrix constructor: mkl_sparse_d_create_csr returned " + std::to_string(stat) );
            }
            
            mreal factor = addToResult ? 1. : 0.;
            
            if( cols > 1 )
            {
                //            tic("MKL sparse matrix-matrix multiplication: cols = " + std::to_string(cols) );
                stat = mkl_sparse_d_mm ( SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, SPARSE_LAYOUT_ROW_MAJOR, input, cols, cols, factor, output, cols );
                if (stat)
                {
                    eprint("in MKLSparseMatrix::Multiply: mkl_sparse_d_mm returned " + std::to_string(stat) );
                }
                //            toc("MKL sparse matrix-matrix multiplication: cols = " + std::to_string(cols) );
            }
            else
            {
                if( cols == 1)
                {
                    //                tic("MKL sparse matrix-vector multiplication" );
                    sparse_status_t stat = mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, input, factor, output );
                    if (stat)
                    {
                        eprint("in MKLSparseMatrix::Multiply: mkl_sparse_d_mv returned " + std::to_string(stat) );
                    }
                    //                toc("MKL sparse matrix-vector multiplication" );
                }
            }
        }
        else
        {
            print("MKLSparseMatrix::Multiply: No nonzeroes found. Doing nothing.");
        }
    }
}; // MKLSparseMatrix


inline mreal mypow ( mreal base, mreal exponent )
{
    // Warning: Use only for positive base! This is basically pow with certain checks and cases deactivated
    return std::exp2( exponent * std::log2(base) );
}

//inline mreal mypow ( mreal base, mreal exponent )
//{
//    // Warning: Use only for positive base! This is basically pow with certain checks and cases deactivated
//    return std::exp( exponent * std::log(base) );
//}

//inline mreal mypow ( mreal base, mreal exponent )
//{
//    return std::pow( base, exponent );
//}

//inline mreal mypow ( mreal base, mreal exponent ){
//    std::exp( exponent * std::log(base) );
//}

} // namespace rsurfaces
