# Stage 1: find the root directory

if (EXISTS "/opt/intel/oneapi/tbb/latest")
  set(TBBROOT_PATH "/opt/intel/oneapi/tbb/latest")
endif (EXISTS "/opt/intel/oneapi/tbb/latest")

# Stage 2: find include path and libraries
  
if (TBBROOT_PATH)
  # root-path found
  
  set(EXPECT_TBB_INCPATH "${TBBROOT_PATH}/include")
  
  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
      set(EXPECT_TBB_LIBPATH "${TBBROOT_PATH}/lib")
  endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  
  set(EXPECT_ICC_LIBPATH "$ENV{ICC_LIBPATH}")
  
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
        if (CMAKE_SIZEOF_VOID_P MATCHES 8)
            if (EXISTS  "${TBBROOT_PATH}/lib/intel64/gcc4.8")
                set(EXPECT_TBB_LIBPATH "${TBBROOT_PATH}/lib/intel64/gcc4.8")
            else (EXISTS  "${TBBROOT_PATH}/lib/intel64/gcc4.8")
                find_path(EXPECT_TBB_LIBPATH, tbb)
            endif (EXISTS  "${TBBROOT_PATH}/lib/intel64/gcc4.8")
        else (CMAKE_SIZEOF_VOID_P MATCHES 8)
            if (EXISTS  "${TBBROOT_PATH}/lib/ia32/gcc4.8")
                set(EXPECT_TBB_LIBPATH "${TBBROOT_PATH}/lib/ia32/gcc4.8")
            else  (EXISTS  "${TBBROOT_PATH}/lib/ia32/gcc4.8")
                find_path(EXPECT_TBB_LIBPATH, tbb)
            endif  (EXISTS  "${TBBROOT_PATH}/lib/ia32/gcc4.8")
        endif (CMAKE_SIZEOF_VOID_P MATCHES 8)
    endif (CMAKE_SYSTEM_NAME MATCHES "Linux")
  
  # set include
  
  if (IS_DIRECTORY ${EXPECT_TBB_INCPATH})
      set(TBB_INCLUDE_DIR ${EXPECT_TBB_INCPATH})
  endif (IS_DIRECTORY ${EXPECT_TBB_INCPATH})
  
  if (IS_DIRECTORY ${EXPECT_TBB_LIBPATH})
    set(TBB_LIBRARY_DIR ${EXPECT_TBB_LIBPATH})
  endif (IS_DIRECTORY ${EXPECT_TBB_LIBPATH})
  
  # find specific library files
    
  find_library(LIB_TBB tbb HINTS ${TBB_LIBRARY_DIR})
else (TBBROOT_PATH)
    find_path(TBB_LIBRARY_DIR tbb)
    find_library(LIB_TBB tbb HINTS ${TBB_LIBRARY_DIR})
    find_path(TBB_INCLUDE_DIR tbb.h)
endif (TBBROOT_PATH)


set(TBB_LIBRARY ${LIB_TBB})

  
# deal with QUIET and REQUIRED argument

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(TBB DEFAULT_MSG 
    TBB_LIBRARY_DIR
    TBB_INCLUDE_DIR)
    
mark_as_advanced(TBB_INCLUDE_DIR)
