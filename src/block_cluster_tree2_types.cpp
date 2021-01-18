#include "block_cluster_tree2_types.h"

namespace rsurfaces
{

    std::deque<std::chrono::time_point<std::chrono::steady_clock>> Timers::time_stack;
    std::chrono::time_point<std::chrono::steady_clock> Timers::start_time;
    std::chrono::time_point<std::chrono::steady_clock> Timers::stop_time;
} // namespace rsurfaces