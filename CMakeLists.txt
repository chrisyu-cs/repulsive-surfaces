cmake_minimum_required(VERSION 3.10.0)

project(gc_polyscope_example)

### Configure output locations
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

# Print the build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
endif()
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

### Configure the compiler
# This is a basic, decent setup that should do something sane on most compilers

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

  # using Clang (linux or apple) or GCC
  message("Using clang/gcc compiler flags")
  SET(BASE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Werror -Xpreprocessor -fopenmp ")
  SET(DISABLED_WARNINGS " -Wno-unknown-warning-option -Wno-pass-failed -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-deprecated-copy -Wno-deprecated-declarations -Wno-missing-braces -Wno-unused-private-field -Wno-range-loop-analysis")
  #SET(BASE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Xpreprocessor -fopenmp -gline-tables-only -gcolumn-info -Rpass=loop-vectorize")
  #SET(DISABLED_WARNINGS " -Wno-unknown-warning-option -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-deprecated-copy -Wno-deprecated-declarations -Wno-missing-braces -Wno-unused-private-field -Wno-range-loop-analysis")
  SET(TRACE_INCLUDES " -H -Wno-error=unused-command-line-argument")
        
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message("Setting clang-specific options")
    SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -ferror-limit=3 -fcolor-diagnostics")
    SET(CMAKE_CXX_FLAGS_DEBUG          "-g3 -fsanitize=address -fno-limit-debug-info")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message("Setting gcc-specific options")
    SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fmax-errors=5")
    SET(CMAKE_CXX_FLAGS_DEBUG          "-g3")
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-maybe-uninitialized -Wno-format-zero-length -Wno-unused-but-set-parameter -Wno-unused-but-set-variable")
  endif()
  
  SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
  SET(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
  message("Using Visual Studio compiler flags")
  set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /W4")
  set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /MP") # parallel build
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4267\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4244\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4305\"")  # ignore truncation on initialization
  SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

  add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
  add_definitions(-DNOMINMAX)
  add_definitions(-D_USE_MATH_DEFINES)
else()
  # unrecognized
  message( FATAL_ERROR "Unrecognized compiler [${CMAKE_CXX_COMPILER_ID}]" )
endif()


# == Deps
add_subdirectory(deps/geometry-central)
add_subdirectory(deps/polyscope)

# == Build our project stuff

set(SRCS 
  src/energy/boundary_length.cpp
  src/energy/coulomb.cpp
  src/energy/implicit_attractor.cpp
  src/energy/implicit_obstacle.cpp
  src/energy/soft_area_constraint.cpp
  src/energy/soft_volume_constraint.cpp
  src/energy/total_area_potential.cpp
  src/energy/total_volume_potential.cpp
  src/energy/tpe_kernel.cpp
  src/energy/squared_error.cpp
  src/energy/willmore_energy.cpp
  src/energy/tpe_multipole_0.cpp
  src/energy/tpe_multipole_nl_0.cpp
  src/energy/tpe_multipole_pr_0.cpp
  src/energy/tpe_barnes_hut_0.cpp
  src/energy/tpe_barnes_hut_pr_0.cpp
  src/energy/tpe_all_pairs.cpp
  src/energy/tpe_all_pairs_pr.cpp
  src/energy/tp_obstacle_multipole_0.cpp
  src/energy/tp_obstacle_multipole_nl_0.cpp
  src/energy/tp_obstacle_multipole_pr_0.cpp
  src/energy/tp_obstacle_barnes_hut_0.cpp
  src/energy/tp_obstacle_barnes_hut_pr_0.cpp
  src/energy/tp_obstacle_all_pairs.cpp
  src/energy/tp_obstacle_all_pairs_pr.cpp
  src/implicit/implicit_surface.cpp
  src/implicit/simple_surfaces.cpp
  src/marchingcubes/CIsoSurface.cpp
  src/marchingcubes/Vectors.cpp
  src/remeshing/dynamic_remesher.cpp
  src/remeshing/remeshing.cpp
  src/sobolev/h1.cpp
  src/sobolev/h2.cpp
  src/sobolev/hs.cpp
  src/sobolev/hs_iterative.cpp
  src/sobolev/hs_ncg.cpp
  src/sobolev/hs_schur.cpp
  src/sobolev/h1_lbfgs.cpp
  src/sobolev/bqn_lbfgs.cpp
  src/sobolev/lbfgs.cpp
  src/sobolev/constraints.cpp
  src/sobolev/constraints/barycenter.cpp
  src/sobolev/constraints/barycenter_components.cpp
  src/sobolev/constraints/scaling.cpp
  src/sobolev/constraints/total_area.cpp
  src/sobolev/constraints/total_volume.cpp
  src/sobolev/constraints/vertex_normal.cpp
  src/sobolev/constraints/vertex_pin.cpp
  src/spatial/convolution_kernel.cpp
  src/optimized_bct_types.cpp
  src/optimized_cluster_tree.cpp
  src/optimized_bct.cpp
  src/fractional_laplacian.cpp
  src/interaction_data.cpp
  src/line_search.cpp
  src/profiler.cpp
  src/matrix_utils.cpp
  src/metric_term.cpp
  src/obj_writer.cpp
  src/scene_file.cpp
  src/surface_derivatives.cpp
  src/surface_flow.cpp
  src/derivative_assembler.cpp
	# add any other source files here
)

set(SRCS1
  src/main.cpp
    # add any other source files here
)

set(SRCS2
  src/main2.cpp
    # add any other source files here
)

find_package(OpenMP REQUIRED)

find_package(MKL REQUIRED)
find_package(TBB REQUIRED)

if(MKL_FOUND)
    message("Intel MKL libraries found: ${MKL_LIBRARY}")
    message("Include directories: ${MKL_INCLUDE_DIR}")
    message("Intel library directories: ${MKL_LIBRARY_DIR}")
    include_directories(${MKL_INCLUDE_DIR})
    #  add_compile_definitions(EIGEN_USE_MKL_ALL)
    link_directories(${MKL_LIBRARY_DIR})
else()
    message(WARNING "Intel MKL libraries not found")
endif()

if(TBB_FOUND)
    include_directories(${TBB_INCLUDE_DIR})
    link_directories(${TBB_LIBRARY_DIR})
    message("Intel TBB libraries: ${TBB_LIBRARY}")
    message("TBB include directories: ${TBB_INCLUDE_DIR}")
    message("TBB library directories: ${TBB_LIBRARY_DIR}")
else()
    message(WARNING "Intel TBB not found")
endif()

add_executable(rsurfaces "${SRCS}" "${SRCS1}")
target_include_directories(rsurfaces PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(rsurfaces geometry-central polyscope OpenMP::OpenMP_CXX)

if(MKL_FOUND)
    target_link_libraries(rsurfaces "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl")
endif()

if(TBB_FOUND)
    target_link_libraries(rsurfaces "-ltbb")
endif()

target_include_directories(rsurfaces PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/libgmultigrid/include")

add_executable(runtest "${SRCS}" "${SRCS2}")
target_include_directories(runtest PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(runtest geometry-central OpenMP::OpenMP_CXX)

if(MKL_FOUND)
    target_link_libraries(runtest "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl")
endif()

if(TBB_FOUND)
    target_link_libraries(runtest "-ltbb")
endif()

target_include_directories(runtest PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/libgmultigrid/include")
