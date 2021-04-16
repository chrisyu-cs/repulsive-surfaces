#include "optimized_bct_types.h"

namespace rsurfaces
{
    std::deque<std::chrono::time_point<std::chrono::steady_clock>> Timers::time_stack;
    std::chrono::time_point<std::chrono::steady_clock> Timers::start_time;
    std::chrono::time_point<std::chrono::steady_clock> Timers::stop_time;

    
    void BalanceWorkLoad( mint job_count, mint * job_acc_costs, mint thread_count, mint * & job_ptr )
    {
        // This function reads in a list job_acc_costs of accumuated costs, then allocates job_ptr as a vector of size thread_count + 1, and writes the work distribution to it.
        // Aasigns threads to consecutive chunks jobs, ..., job_ptr[k+1]-1 of jobs.
        // Uses a binary search to find the chunk boundaries.
        // The cost of the i-th job is job_acc_costs[i+1] - job_acc_costs[i].
        // The cost of the k-th thread goes from job no job_ptr[k] to job no job_ptr[k+1] (as always in C/C++, job_ptr[k+1] points _after_ the last job.
        
        ptic("BalanceWorkLoad");
        safe_alloc( job_ptr, thread_count + 1);
        job_ptr[0] = 0;
        job_ptr[ thread_count ] = job_count;
        
        mint naive_chunk_size = (job_count + thread_count - 1) / thread_count;
        
//        auto naive_job_ptr = std::vector<mint>(job_count + 1);
//        for( mint thread = 0; thread < thread_count - 1; ++ thread)
//        {
//            naive_job_ptr[thread+1] = std::min( job_count, naive_chunk_size * (thread+1) );
//        }
        
        mint total_cost = job_acc_costs[job_count];
        mint per_thread_cost = (total_cost + thread_count - 1) / thread_count;
        
        
//         TODO: There is quite a lot false sharing in this loop...
        // binary search for best work load distribution
        #pragma omp parallel for num_threads(thread_count) schedule( static, 1 )
        for( mint thread = 0; thread < thread_count - 1; ++thread)
        {
//            std::cout << "\n #### thread = " << thread << std::endl;
            // each thread (other than the last one) is require to have at least this accumulated cost
            mint target = std::min( total_cost, per_thread_cost * (thread + 1) );
            mint pos;
            // find an index a such that b_row_acc_costs[ a ] < target;
            // taking naive_chunk_size * thread as initial guess, because that might be nearly correct for costs that are evenly distributed over the block rows
            pos = thread + 1;
            mint a = std::min(job_count, naive_chunk_size * pos);
            while( job_acc_costs[ a ] >= target )
            {
                --pos;
                a = std::min(job_count, naive_chunk_size * pos);
            };
            
            // find an index  b such that b_row_acc_costs[ b ] >= target;
            // taking naive_chunk_size * (thread + 1) as initial guess, because that might be nearly correct for costs that are evenly distributed over the block rows
            pos = thread + 1;
            mint b = std::min(job_count, naive_chunk_size * pos);
            while( job_acc_costs[ b ] < target && b < job_count)
            {
                ++pos;
                b = std::min(job_count, naive_chunk_size * pos);
            };

            // binary search until
            mint c;
            while( b > a + 1 )
            {
                c = a + (b-a)/2;
                if( job_acc_costs[c] > target )
                {
                    b = c;
                }
                else
                {
                    a = c;
                }
            }
            job_ptr[thread + 1] = b;
        }
        
//        valprint("job_ptr", job_ptr, job_ptr + thread_count + 1);
//
//        std::cout << "workloads = [ ";
//        for( mint thread = 0; thread < thread_count - 1; ++thread)
//        {
//            std::cout << job_acc_costs[job_ptr[thread+1]] -  job_acc_costs[job_ptr[thread]] << ", ";
//        }
//        std::cout << job_acc_costs[job_ptr[thread_count]] -  job_acc_costs[job_ptr[thread_count-1]];
//        std::cout << " ]" << std::endl;
        
        ptoc("BalanceWorkLoad");
    } // BalanceWorkLoad
    
} // namespace rsurfaces
