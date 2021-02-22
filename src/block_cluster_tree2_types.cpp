#include "block_cluster_tree2_types.h"

namespace rsurfaces
{
    mreal * mreal_alloc(size_t size)
    {
        return (mreal *) mkl_malloc ( size * sizeof(mreal), ALIGN );
    }

    mreal * mreal_alloc(size_t size, mreal init)
    {
        mreal * ptr = mreal_alloc(size);
        #pragma omp simd aligned( ptr : ALIGN )
        for( size_t i = 0; i < size; ++i )
        {
            ptr[i] = init;
        }
        return ptr;
    }

    void  mreal_free(mreal * ptr)
    {
        if( ptr ){ mkl_free(ptr); }
    }

    mint * mint_alloc(size_t size)
    {
        return (mint * ) mkl_malloc ( size * sizeof(mint), size );
    }

    mint * mint_alloc(size_t size, mint init)
    {
        mint * ptr = mint_alloc(size);
        #pragma omp simd aligned( ptr : ALIGN )
        for( size_t i = 0; i < size; ++i )
        {
            ptr[i] = init;
        }
        return ptr;
    }

    void  mint_free(mint * ptr)
    {
        if( ptr ){ mkl_free(ptr); }
    }

    std::deque<std::chrono::time_point<std::chrono::steady_clock>> Timers::time_stack;
    std::chrono::time_point<std::chrono::steady_clock> Timers::start_time;
    std::chrono::time_point<std::chrono::steady_clock> Timers::stop_time;
} // namespace rsurfaces
