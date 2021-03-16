#include "profiler.h"

namespace rsurfaces
{
    std::ofstream Profiler::os ( "./Profile.tsv");
    std::chrono::time_point<std::chrono::steady_clock> Profiler::init_time = std::chrono::steady_clock::now();
    std::deque<std::chrono::time_point<std::chrono::steady_clock>> Profiler::time_stack;
    std::deque<std::string> Profiler::tag_stack(1,"root");
    std::deque<int> Profiler::parent_stack (1, -0);
    std::deque<int> Profiler::id_stack (1,0);
    int Profiler::id_counter = 0;
} // namespace rsurfaces
