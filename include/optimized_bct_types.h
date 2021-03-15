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
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include "bct_kernel_type.h"

#define restrict __restrict
#define ALIGN 32

namespace rsurfaces
{

    // "In know only two types: integers and doubles..."
    typedef MKL_INT mint; // "machine integer" -- ensuring that we use the integer type requested by MKL. I find "MKL_INT" a bit clunky, though.
    typedef double mreal; // "machine real"

    mreal * mreal_alloc(size_t size);
    mreal * mreal_alloc(size_t size, mreal init);
    void  mreal_free(mreal * ptr);
    mint * mint_alloc(size_t size);
    mint * mint_alloc(size_t size, mint init);
    mint * mint_iota(size_t size, mint step = 1);
    void  mint_free(mint * ptr);


    typedef Eigen::SparseMatrix<mreal, Eigen::RowMajor, mint> EigenMatrixCSR;
    typedef Eigen::SparseMatrix<mreal> EigenMatrixCSC;

    typedef Eigen::Matrix<mreal, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixRM;
    typedef Eigen::MatrixXd EigenMatrixCM;

    class Timers
    {
        public:
        static std::deque<std::chrono::time_point<std::chrono::steady_clock>> time_stack;
        static std::chrono::time_point<std::chrono::steady_clock> start_time;
        static std::chrono::time_point<std::chrono::steady_clock> stop_time;
    };

    inline void print(std::string s)
    {
        std::cout << (std::string( 2 * (Timers::time_stack.size() + 1), ' ') + s) << std::endl;
    }

    inline void valprint(std::string s, mint val)
    {
        std::cout << (std::string( 2 * (Timers::time_stack.size() + 1), ' ') + s) << " = " << val <<  std::endl;
    }

    inline void valprint(std::string s, mreal val)
    {
        std::cout << (std::string( 2 * (Timers::time_stack.size() + 1), ' ') + s) << " = " << val <<  std::endl;
    }

    inline void eprint(std::string s)
    {
        std::cout << (std::string( 2 * (Timers::time_stack.size() + 1), ' ') + "ERROR: " + s) << std::endl;
    }

    inline void wprint(std::string s)
    {
        std::cout << (std::string( 2 * (Timers::time_stack.size() + 1), ' ') + "WARNING: " + s) << std::endl;
    }

    inline void tic(std::string s)
    {
        Timers::time_stack.push_back(std::chrono::steady_clock::now());
        std::cout << (std::string( 2 * Timers::time_stack.size(), ' ') + s + "...") << std::endl;
    }

    inline mreal toc(std::string s)
    {
        if (!Timers::time_stack.empty())
        {
            auto start_time = Timers::time_stack.back();
            auto stop_time = std::chrono::steady_clock::now();
            mreal duration = std::chrono::duration<double>(stop_time - start_time).count();
            std::cout << (std::string( 2 * Timers::time_stack.size(), ' ') + std::to_string(duration) + " s.") << std::endl;
            Timers::time_stack.pop_back();
            return duration;
        }
        else
        {
            std::cout << ("Unmatched toc detected. Label =  " + s) << std::endl;
            return 0.;
        }
    }

    inline void tic()
    {
        Timers::time_stack.push_back(std::chrono::steady_clock::now());
    }

    inline mreal toc()
    {
        if (!Timers::time_stack.empty())
        {
            auto start_time = Timers::time_stack.back();
            auto stop_time = std::chrono::steady_clock::now();
            mreal duration = std::chrono::duration<double>(stop_time - start_time).count();
            Timers::time_stack.pop_back();
            return duration;
        }
        else
        {
            std::cout << ("Unmatched toc detected.") << std::endl;
            return 0.;
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


    struct PardisoData
    {
        mint n = 0;
        mint mtype = 11;                   /* Matrix type */
        mint * restrict perm = nullptr;       /* Permutation */
        mint * restrict iparm = nullptr;      /* Integer parameter array for controlling pardiso */
        A_Vector<void*> pt;                /* Pointer used internally by pardiso to store its data */
    
        bool symfactorized = false;
        bool numfactorized = false;
    
        PardisoData(){};
    
        ~PardisoData(){
            mint_free(perm);
            mint_free(iparm);
        };
        
        // Copy constructor
        PardisoData( PardisoData const & P )
        {
            n = P.n;
            mtype = P.mtype;
            pt = P.pt;
            symfactorized = P.symfactorized;
            numfactorized = P.numfactorized;
            if( P.perm )
            {
                const mint * const restrict ptr = P.perm;
                perm = mint_alloc(n);
                #pragma omp simd aligned( perm, ptr : ALIGN)
                for( mint i = 0; i < n; ++i )
                {
                    perm[i] = ptr[i];
                }
            }
            if( P.iparm )
            {
                const mint * const restrict ptr = P.iparm;
                iparm = mint_alloc(64);
                #pragma omp simd aligned( iparm, ptr : ALIGN)
                for( mint i = 0; i < 64; ++i )
                {
                    iparm[i] = ptr[i];
                }
            }
        }
        
        // Move constructor
        PardisoData( PardisoData && P )
        {
            n = std::move(P.n);
            mtype = std::move(P.mtype);
            pt = std::move(P.pt);
            symfactorized = std::move(P.symfactorized);
            numfactorized = std::move(P.numfactorized);
            
            perm = std::move(P.perm);
            iparm = std::move(P.iparm);
            
        }
        
        PardisoData &operator=(PardisoData P)
        {
            P.swap(*this);
            return *this;
        }
    private:
        void swap(PardisoData &P) throw()
        {
            std::swap(this->n, P.n);
            std::swap(this->mtype, P.mtype);
            std::swap(this->pt, P.pt);
            std::swap(this->symfactorized, P.symfactorized);
            std::swap(this->numfactorized, P.numfactorized);

            std::swap(this->perm, P.perm);
            std::swap(this->iparm, P.iparm);

        }
    }; // PardisoData

    struct MKLSparseMatrix //A container to hold generic sparse array data and to perform MKL matrix-matrix-multiplication routines
    {
        mint m = 0;
        mint n = 0;
        mint nnz = 0;
        mint  * restrict outer = nullptr;
        mint  * restrict inner = nullptr;
        mreal * restrict values = nullptr;
        PardisoData P;
    
        matrix_descr descr;
    
        //    MKLSparseMatrix( SparseMatrix A ) : MKLSparseMatrix( A.rows(), A.cols(), A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr() ){}; //dunno why this does not work...
    
        MKLSparseMatrix( const mint m_, const mint n_, const mint nnz_ )
        {
//            print("MKLSparseMatrix( const mint m_, const mint n_, const mint nnz_ )");
            m = m_;
            n = n_;
            nnz = nnz_;
            P.n = n;
        
            outer  =  mint_alloc( m + 1 );
            inner  =  mint_alloc( nnz );
            values = mreal_alloc( nnz );
            outer[0] = 0;
            outer[m_] = nnz;
        
            descr.type = SPARSE_MATRIX_TYPE_GENERAL;
            descr.diag = SPARSE_DIAG_NON_UNIT;
            P.mtype = 11;
        };
    
        MKLSparseMatrix( const mint m_, const mint n_, mint * outer_, mint * inner_, mreal * values_ )
        {
//            print("MKLSparseMatrix( const mint m_, const mint n_, mint * outer_, mint * inner_, mreal * values_ )");
            m = m_;
            n = n_;
            nnz = outer_[m];
            P.n = n;
            
            outer  =  mint_alloc( m + 1 );
            inner  =  mint_alloc( nnz );
            values = mreal_alloc( nnz );
        
            #pragma omp simd aligned( outer : ALIGN )
            for( mint i = 0; i < m+1; ++i)
            {
                outer[i] = outer_[i];
            }
        
            #pragma omp simd aligned( inner, values : ALIGN )
            for( mint i = 0; i < nnz; ++i)
            {
                inner[i] = inner_[i];
                values[i] = values_[i];
            }
        
            descr.type = SPARSE_MATRIX_TYPE_GENERAL;
            descr.diag = SPARSE_DIAG_NON_UNIT;
            P.mtype = 11;
        
            //        vprint("m",m);
            //        vprint("n",n);
            //        vprint("nnz",nnz);
        };
    
        MKLSparseMatrix( const mint m_, const mint n_, mint * outer_B, mint * outer_E, mint * inner_, mreal * values_ )
        {
//            print("MKLSparseMatrix( const mint m_, const mint n_, mint * outer_B, mint * outer_E, mint * inner_, mreal * values_ )");
            m = m_;
            n = n_;
            nnz = outer_B[m];
            P.n = n;
            
            if(outer_B[0])
            {
                eprint("in MKLSparseMatrix: outer_B[0] != 0.");
            }
        
            outer  =  mint_alloc( m + 1 );
            inner  =  mint_alloc( nnz );
            values = mreal_alloc( nnz );
        
            outer[0] = 0;
        
            #pragma omp simd aligned( outer : ALIGN )
            for( mint i = 0; i < m; ++i)
            {
                outer[i+1] = outer_E[i];
            }
        
            #pragma omp simd aligned( inner, values : ALIGN )
            for( mint i = 0; i < nnz; ++i)
            {
                inner[i] = inner_[i];
                values[i] = values_[i];
            }
        
            descr.type = SPARSE_MATRIX_TYPE_GENERAL;
            descr.diag = SPARSE_DIAG_NON_UNIT;
            P.mtype = 11;
        };
    

        // Copy constructor
        MKLSparseMatrix( MKLSparseMatrix const & B )
        {
//            print("MKLSparseMatrix copy constructor");
            m = B.m;
            n = B.n;
            nnz = B.nnz;
            
            P = B.P;
            
            if(B.outer[0])
            {
                eprint("in MKLSparseMatrix &operator=(MKLSparseMatrix const &B): B.outer[0] != 0.");
            }
            
            outer  =  mint_alloc( m + 1 );
            inner  =  mint_alloc( nnz );
            values = mreal_alloc( nnz );
            
            #pragma omp simd aligned( outer : ALIGN )
            for( mint i = 0; i <= m; ++i)
            {
                outer[i] = B.outer[i];
            }
            
            #pragma omp simd aligned( inner, values : ALIGN )
            for( mint i = 0; i < nnz; ++i)
            {
                inner[i] = B.inner[i];
                values[i] = B.values[i];
            }
            
            descr.type = B.descr.type;
            descr.diag = B.descr.diag;
        }
        
        // Move constructor
        MKLSparseMatrix( MKLSparseMatrix && B )
        {
//            print("MKLSparseMatrix move constructor");
            m = B.m;
            n = B.n;
            nnz = B.nnz;
            P = std::move(B.P);
            
            outer = std::move(B.outer);
            inner = std::move(B.inner);
            values = std::move(B.values);
            
            descr = std::move(B.descr);
            
        }
        
        // copy-and-swap idiom
        MKLSparseMatrix &operator=(MKLSparseMatrix B)
        {
//            print("MKLSparseMatrix copy-and-swap");
            B.swap(*this);
            return *this;
        }
        
        MKLSparseMatrix(){
            descr.type = SPARSE_MATRIX_TYPE_GENERAL;
            descr.diag = SPARSE_DIAG_NON_UNIT;
        };
    
        ~MKLSparseMatrix(){
            
//            print("~MKLSparseMatrix()");
//            PrintStats();
            if( P.symfactorized || P.numfactorized )
            {
                mint phase = -1;
                mreal ddum = 0.;                /* double dummy pointer */
                mint mnum = 1;                  /* Which factorization to use */
                mint error = 0;                 /* Error flag */
                mint nrhs = 1;                  /* Number of right hand sides */
                mint msglvl = 0;                /* Do not print statistical information to file */
                mint maxfct = 1;                /* Maximum number of numerical factorizations */
                pardiso (P.pt.data(), &maxfct, &mnum, &P.mtype, &phase, &n, values, outer, inner, P.perm, &nrhs, P.iparm, &msglvl, &ddum, &ddum, &error);
            }
            
            mint_free( outer );
            mint_free( inner );
            mreal_free( values );
        };
        
        
        void PrintStats()
        {
            print("##################################");
            valprint("m", m);
            valprint("n", n);
            valprint("nnz", nnz);
            valprint("P.n", P.n);
//            std::cout << "descr = " << descr << std::endl;
            
            if( outer )
            {
                valprint("outer[0]", outer[0]);
                valprint("outer[m]", outer[m]);
            }
            if( inner )
            {
                valprint("inner[0]", inner[0]);
                valprint("inner[nnz]", inner[nnz-1]);
            }
            if( values )
            {
                valprint("values[0]", values[0]);
                valprint("values[nnz]", values[nnz-1]);
            }
            print("##################################");
        }
        
        void Check()
        {
            print(" ### MKLSparseMatrix::Check ### ");
            valprint("m", m);
            valprint("n", n);
            valprint("nnz", nnz);
            
            bool failed = false;
            
            
            if( outer )
            {
                if( outer[0] )
                {
                    eprint("outer[0] != 0");
                    valprint("outer[0]", outer[0]);
                    failed = true;
                }

                if( outer[m] != nnz )
                {
                    eprint("outer[m] != nnz");
                    failed = true;
                }
            }
            else
            {
                wprint("outer not initilized.");
            }
            
            if( inner )
            {
                valprint("inner[0]", inner[0]);
                valprint("inner[nnz]", inner[nnz-1]);
            }
            else
            {
                wprint("inner not initilized.");
            }
            if( values )
            {
                valprint("values[0]", values[0]);
                valprint("values[nnz]", values[nnz-1]);
            }
            else
            {
                wprint("values not initilized.");
            }
            

            if( outer && inner )
            {
                for( mint i = 0; i < m; ++ i)
                {
                    if( failed )
                    {
                        break;
                    }
                    for( mint k = outer[i]; k< outer[i+1]; ++k )
                    {
                        mint j = inner[k];
                        
                        if( j<0 || j >= n)
                        {
                            eprint("inner[" + std::to_string(k) + "] is out of bounds.");
                            valprint("i", i);
                            valprint("j = inner[k]", j);
                            failed = true;
                            break;
                        }
                    }
                }
            }
            
            print(" ### MKLSparseMatrix::Check finished ### ");
        }
    
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
            
                sparse_matrix_t A = nullptr;
            
                stat = mkl_sparse_d_create_csr ( &A, SPARSE_INDEX_BASE_ZERO, m, n, outer, outer + 1, inner, values );
                if (stat)
                {
                    eprint(" in MKLSparseMatrix::Multiply: mkl_sparse_d_create_csr returned " + std::to_string(stat) );
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
                
                stat = mkl_sparse_destroy(A);
                if (stat)
                {
                    eprint("mkl_sparse_destroy returned stat = " + std::to_string(stat) );
                }
            }
            else
            {
                wprint("MKLSparseMatrix::Multiply: No nonzeroes found. Doing nothing.");
            }
        }
    
        void Multiply( MKLSparseMatrix & B, MKLSparseMatrix & C)
        {
//            print("void Multiply( MKLSparseMatrix & B, MKLSparseMatrix & C)");
            
            sparse_status_t stat;
            sparse_matrix_t csrA = nullptr;
            sparse_matrix_t csrB = nullptr;
            sparse_matrix_t csrC = nullptr;

            stat = mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, m, n, outer, outer + 1, inner, values );
            if (stat)
            {
                eprint("in MKLSparseMatrix::Multiply: mkl_sparse_d_create_csr returned " + std::to_string(stat) );
            }

            stat = mkl_sparse_d_create_csr(&csrB, SPARSE_INDEX_BASE_ZERO, B.m, B.n, B.outer, B.outer +1 , B.inner, B.values );
            if (stat)
            {
                eprint("in MKLSparseMatrix::Multiply: mkl_sparse_d_create_csr returned " + std::to_string(stat) );
            }

            stat = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrC);
            if (stat)
            {
                eprint("in MKLSparseMatrix::Multiply: mkl_sparse_spmm returned " + std::to_string(stat) );
            }

            mint rows_C;
            mint cols_C;
            
            mint  * inner_C  = nullptr;
            mint  * outerB_C = nullptr;
            mint  * outerE_C = nullptr;
            mreal * values_C = nullptr;
            
            sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;

            stat = mkl_sparse_d_export_csr( csrC, &indexing, &rows_C, &cols_C, &outerB_C, &outerE_C, &inner_C, &values_C );
            if (stat)
            {
                eprint("in MKLSparseMatrix::Multiply: mkl_sparse_d_export_csr returned " + std::to_string(stat) );
            }

            C = MKLSparseMatrix( rows_C, cols_C, outerB_C, outerE_C, inner_C, values_C ); // Copy!

            mkl_sparse_destroy(csrA);
            mkl_sparse_destroy(csrB);
            mkl_sparse_destroy(csrC);
        }
        
        
        void Transpose( MKLSparseMatrix & AT)
        {
            
            sparse_status_t stat;
            sparse_matrix_t csrA = nullptr;
            sparse_matrix_t csrAT = nullptr;
            
            stat = mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, m, n, outer, outer + 1, inner, values );
            if (stat)
            {
                eprint("in MKLSparseMatrix::Transpose: mkl_sparse_d_create_csr returned " + std::to_string(stat) );
            }
            stat = mkl_sparse_convert_csr (csrA, SPARSE_OPERATION_TRANSPOSE, &csrAT);
            if (stat)
            {
                eprint("in MKLSparseMatrix::Transpose: mkl_sparse_convert_csr returned " + std::to_string(stat) );
            }
            mint rows_AT;
            mint cols_AT;
            
            mint  * inner_AT  = nullptr;
            mint  * outerB_AT = nullptr;
            mint  * outerE_AT = nullptr;
            mreal * values_AT = nullptr;
            
            sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
            
            stat = mkl_sparse_d_export_csr( csrAT, &indexing, &rows_AT, &cols_AT, &outerB_AT, &outerE_AT, &inner_AT, &values_AT ); // It's not logical to swap rows_AT and cols_AT...
            if (stat)
            {
                eprint("in MKLSparseMatrix::Transpose: mkl_sparse_d_export_csr returned " + std::to_string(stat) );
            }
            
            AT = MKLSparseMatrix( rows_AT, cols_AT, outerB_AT, outerE_AT, inner_AT, values_AT ); // Copy!
            
            mkl_sparse_destroy(csrA);
            mkl_sparse_destroy(csrAT);
        }
        
        
        mint FactorizeSymbolically()
        {
            if( m != n)
            {
                eprint("Matrix is not square, so it cannot be factorized symbolically.");
                return 1;
            }
            else
            {
                P.pt = A_Vector<void*>(64);
                P.iparm = mint_alloc(64);
                P.perm = mint_alloc(m);
                
                for ( mint i = 0; i < m; ++i )
                {
                    P.perm[i] = i;
                }
                
                P.iparm[0] = 1;               /* No solver default */
                P.iparm[1] = 3;               /* parallel version of nested disection */
                P.iparm[3] = 0;               /* No iterative-direct algorithm */
                P.iparm[4] = 2;               /* Write fill-in reducing permutation to perm */
                P.iparm[5] = 0;               /* Write solution into x */
                if( P.mtype == 11 )
                {
                    P.iparm[9] = 13;          /* Perturb the pivot elements with 1E-iparm[9] */
                }
                else
                {
                    P.iparm[9] = 8;           /* Perturb the pivot elements with 1E-iparm[9] */
                }
                if( (P.mtype==2) || (P.mtype==-2) )
                {
                    P.iparm[10] = 0;          /* Disable scaling. Because it is slow.*/
                    P.iparm[12] = 0;          /* Disable matching. Because it is slow.*/
                }
                else
                {
                    P.iparm[10] = 1;          /* Enable scaling. Default for nonsymmetric matrices. Good for indefinite symmetric matrices */
                    P.iparm[12] = 1;          /* Enable matching. Default for nonsymmetric matrices. Good for indefinite symmetric matrices */
                }
                P.iparm[17] = -1;             /* Report number of nonzeros in the factor LU */
                P.iparm[18] = 0;              /* Do not compute Mflops for LU factorization (because it is not for free) */
                
                P.iparm[20] = 1;              /* Bunch-Kaufman pivoting */
                P.iparm[34] = 1;              /* 0-based indexing */
                
                mint phase = 11;
                
                mreal ddum = 0.;                /* double dummy pointer */
                mint mnum = 1;                  /* Which factorization to use */
                mint error = 0;                 /* Error flag */
                mint nrhs = 1;                  /* Number of right hand sides */
                mint msglvl = 0;                /* Do not print statistical information to file */
                mint maxfct = 1;                /* Maximum number of numerical factorizations */
                
                pardiso( P.pt.data(), &maxfct, &mnum, &P.mtype, &phase, &n, values, outer, inner, P.perm, &nrhs, P.iparm, &msglvl, &ddum, &ddum, &error );
                if(error!=0)
                {
                    P.symfactorized = false;
                    eprint("Pardiso reported an error in symbolic factorization: error = " + std::to_string(error) );
                }
                else
                {
                    P.symfactorized = true;
                }
                return error;
            }
        } // FactorizeSymbolically
        
        mint FactorizeNumerically()
        {
            if(!P.symfactorized)
            {
                FactorizeSymbolically();
            }
            mint phase = 22;
            
            mreal ddum = 0.;                /* double dummy pointer */
            mint mnum = 1;                  /* Which factorization to use */
            mint error = 0;                 /* Error flag */
            mint nrhs = 1;                  /* Number of right hand sides */
            mint msglvl = 0;                /* Do not print statistical information to file */
            mint maxfct = 1;                /* Maximum number of numerical factorizations */
            pardiso( P.pt.data(), &maxfct, &mnum, &P.mtype, &phase, &n, values, outer, inner, P.perm, &nrhs, P.iparm, &msglvl, &ddum, &ddum, &error );
            if(error!=0)
            {
                P.numfactorized = false;
                eprint("Pardiso reported an error in numeric factorization: error = " + std::to_string(error) );
            }
            else
            {
                P.numfactorized = true;
            }
            return error;
        } // FactorizeNumerically
        
        mint LinearSolve(mreal * b, mreal * x, bool transposed = false)
        {
            // solves A * x = b
            if(!P.numfactorized)
            {
                FactorizeNumerically();
            }
            mint phase = 33;
            P.iparm[11] = transposed ? 1 : 0;
            
            mint mnum = 1;                  /* Which factorization to use */
            mint error = 0;                 /* Error flag */
            mint nrhs = 1;                  /* Number of right hand sides */
            mint msglvl = 0;                /* Do not print statistical information to file */
            mint maxfct = 1;                /* Maximum number of numerical factorizations */
            pardiso( P.pt.data(), &maxfct, &mnum, &P.mtype, &phase, &n, values, outer, inner, P.perm, &nrhs, P.iparm, &msglvl, b, x, &error );
            if(error!=0)
            {
                eprint("Pardiso reported an error in solving phase: error = " + std::to_string(error) );
            }
            P.iparm[11] = 0;
            return error;
        } // LinearSolve
        
        // TODO: Currently untested whether multiple right hand sides are handled correctly here.
        mint LinearSolveMatrix(mreal * B, mreal * X, mint cols, bool transposed = false)
        {
            // solves A * X = B
            if(!P.numfactorized)
            {
                FactorizeNumerically();
            }
            mint phase = 33;
            P.iparm[11] = transposed ? 1 : 0;
            
            mint mnum = 1;                  /* Which factorization to use */
            mint error = 0;                 /* Error flag */
            mint nrhs = cols;               /* Number of right hand sides */
            mint msglvl = 0;                /* Do not print statistical information to file */
            mint maxfct = 1;                /* Maximum number of numerical factorizations */
            pardiso(P.pt.data(), &maxfct, &mnum, &P.mtype, &phase, &n, values, outer, inner, P.perm, &nrhs, P.iparm, &msglvl, B, X, &error );
            if(error!=0)
            {
                eprint("Pardiso reported an error in solving phase: error = " + std::to_string( error) );
            }
            P.iparm[11] = 0;
            return error;
        } // LinearSolveMatrix
        
        private:
        void swap(MKLSparseMatrix &B) throw()
        {
            std::swap(this->m, B.m);
            std::swap(this->n, B.n);
            std::swap(this->nnz, B.nnz);
            
            std::swap(this->outer, B.outer);
            std::swap(this->inner, B.inner);
            std::swap(this->values, B.values);
            std::swap(this->P, B.P);
            std::swap(this->descr, B.descr);
        }
        
    }; // MKLSparseMatrix



    #pragma omp declare simd
    inline mreal mypow ( mreal base, mreal exponent )
    {
        // Warning: Use only for positive base! This is basically pow with certain checks and cases deactivated
        return std::exp2( exponent * std::log2(base) );
    } // mypow

//    #pragma omp declare simd
//    inline mreal intpow(mreal base, mint exponent)
//    {
//        mreal r = 1.;
//        mreal x = base;
//        mint  k = abs(exponent);
//
//        while( k > 0)
//        {
//            if( k % 2 )
//            {
//                r *= x;
//            }
//            x *= x;
//            k /= 2;
//        }
//        return exponent >= 0 ? r : 1./r;
//    } // intpow

//#pragma omp declare simd
//inline mreal intpow(const mreal base, const double exponent)
//{
//    mreal r = 1.;
//    mint last = std::round(std::floor(abs(exponent)));
//    for( mint k = 0; k < last; ++k )
//    {
//        r *= base;
//    }
//    return exponent>=0. ? r : 1./r;
//} // intpow

    #pragma omp declare simd
    inline mreal mypow(mreal base, mint exponent)
    {
        mreal b2, b3, b4, b6;
        if( exponent >= 0)
        {
            switch (exponent) {
                case 0: return 1.;
                case 1: return base;
                case 2: return base * base;
                case 3: return base * base * base;
                case 4:
                    b2 = base * base;
                    return b2 * b2;
                case 5:
                    b2 = base * base;
                    return b2 * b2 * base;
                case 6:
                    b2 = base * base;
                    return b2 * b2 * b2;
                case 7:
                    b2 = base * base;
                    b4 = b2 * b2;
                    return b4 * b2 * base;
                case 8:
                    b2 = base * base;
                    b4 = b2 * b2;
                    return b4 * b4;
                case 9:
                    b2 = base * base;
                    b4 = b2 * b2;
                    return b4 * b4 * base;
                case 10:
                    b2 = base * base;
                    b4 = b2 * b2;
                    return b4 * b4 * b2;
                case 11:
                    b2 = base * base;
                    b4 = b2 * b2;
                    return b4 * b4 * b2 * base;
                case 12:
                    b2 = base * base;
                    b4 = b2 * b2;
                    return b4 * b4 * b4;

                default:
                    return mypow(base, exponent);
            }
        }
        else
        {
            return 1./mypow(base, -exponent);
        }
    } // mypow

    #pragma omp declare simd
    inline mreal mymax(const mreal & a, const mreal & b)
    {
        return fmax(a,b);
    }

    #pragma omp declare simd
    inline mreal mymin(const mreal & a, const mreal & b)
    {
        return fmin(a,b);
    }


} // namespace rsurfaces
