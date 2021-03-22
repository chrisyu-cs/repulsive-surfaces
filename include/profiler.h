#pragma once

#include <algorithm>
#include <vector>
#include <deque>
#include <iterator>
#include <memory>
#include <unistd.h>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include <omp.h>

namespace rsurfaces
{
    
    class Profiler
    {
        public:
        static int id_counter;
        static std::ofstream os;
        static std::chrono::time_point<std::chrono::steady_clock> init_time;
        static std::deque<std::chrono::time_point<std::chrono::steady_clock>> time_stack;
        static std::deque<int> id_stack;
        static std::deque<std::string> tag_stack;
        static std::deque<int> parent_stack;
    };
    
    
//    inline void ptic(std::string tag){};
//    inline void ptoc(std::string tag){};
//    inline void ClearProfile(std::string filename){}
    
    inline void ptic(std::string tag)
    {
        Profiler::time_stack.push_back(std::chrono::steady_clock::now());
        Profiler::parent_stack.push_back(Profiler::id_stack.back());
        Profiler::tag_stack.push_back(tag);
        Profiler::id_stack.push_back(++Profiler::id_counter);
        
    }

    inline void ptoc(std::string tag)
    {
        if( !Profiler::time_stack.empty() || tag != Profiler::tag_stack.back() )
        {
            double start_time = std::chrono::duration<double>( Profiler::time_stack.back()     - Profiler::init_time ).count();
            double stop_time  = std::chrono::duration<double>( std::chrono::steady_clock::now() - Profiler::init_time ).count();

            Profiler::os
                << Profiler::id_stack.back() <<  "\t"
                << Profiler::tag_stack.back() << "\t"
                << Profiler::parent_stack.back() << "\t"
                << start_time << "\t"
                << stop_time << "\t"
                << stop_time-start_time << "\t"
                << Profiler::tag_stack.size()-1
                << std::endl;
            
            Profiler::id_stack.pop_back();
            Profiler::time_stack.pop_back();
            Profiler::tag_stack.pop_back();
            Profiler::parent_stack.pop_back();
        }
        else
        {
            std::cout << ("Unmatched ptoc detected. Label =  " + tag) << std::endl;
        }
    }

    
    inline void ClearProfile(std::string filename)
    {
        Profiler::os.close();
        Profiler::os.open(filename);
//        Profiler::os << "ID" << "\t" << "Tag" << "\t" << "From" << "\t" << "Tic" << "\t" << "Toc" << "\t" << "Duration" << "\t" << "Depth" << std::endl;
        Profiler::init_time = std::chrono::steady_clock::now();
        Profiler::time_stack.clear();
        Profiler::parent_stack.push_back(0.);
        Profiler::tag_stack.clear();
        Profiler::tag_stack.push_back("root");
        Profiler::parent_stack.clear();
        Profiler::parent_stack.push_back(-1);
        Profiler::id_stack.clear();
        Profiler::id_stack.push_back(0);
        Profiler::id_counter = 0;
    }

} // namespace rsurfaces
