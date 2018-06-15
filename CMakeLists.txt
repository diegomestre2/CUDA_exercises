
# CMakeLists.txt for the Cuda_exercises project
#
# Notes:
#
# To use a different compiler with CMake, run it as follows:
#
# cmake -D CMAKE_C_COMPILER=/path/to/your/cc -D CMAKE_CXX_COMPILER=/path/to/your/c++ your_project_dir
#
#
cmake_minimum_required (VERSION 3.1)

# My own cmake Modules
# 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
#
#
set(CMAKE_VERBOSE_MAKEFILE OFF)
# -----------------------------------
# Project name, version & build type
# -----------------------------------



if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
       Ex: 
           mkdir build/debug
           cd build/debug
           cmake ..
")
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # This is the top-level project.
	PROJECT (CUDA)

#	# No versioning for now
#
#	set(PROJECT_MAJOR_VERSION 0)
#	set(PROJECT_MINOR_VERSION 1)
#	set(PROJECT_PATCH_VERSION 0)
#	set(PROJECT_VERSION ${PROJECT_MAJOR_VERSION}.${PROJECT_MINOR_VERSION}.${PROJECT_PATCH_VERSION})
endif()

set(LIBS
        #gtest
        #gtest_main
        pthread
        #numa
        cudart
        )

# ----------------------------
# General C/C++ build settings
# ----------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall " )
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -Og -g -march=native")

# ------------------------
# Looking for Dependencies
# ------------------------

# NVidia CUB
# git clone git@github.com:NVlabs/cub.git
find_package(CUB 1.5.0 REQUIRED)
if(CUB_FOUND)
    include_directories(${CUB_INCLUDE_DIRS})
    message(STATUS "CUB FOUND: ${CUB_INCLUDE_DIRS}")
else()
    # Database Template Library (DTL)
    # git clone https://github.com/NVlabs/cub.git cub
    message(STATUS "CUB NOT FOUND. Including it as submodule...")
    include_directories(../cub/cub/)
endif()

# CUDAAPIWrappers
find_package(CUDAAPIWrappers REQUIRED)
include_directories(${CUDA_API_WRAPPERS_INCLUDE_DIRS})
# but what about the library search path?
link_libraries(${CUDA_API_WRAPPERS_LIBS} "-lcudadevrt")
#add_subdirectory(../cub/cub cub)

include_directories(SYSTEM ../)
include_directories("include")

# # CUDA

find_package(CUDA 9.1 REQUIRED)
include(HandleCUDAComputeCapability)
# avoid implicitly synchronization with other streams
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --default-stream per-thread)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --restrict --std=c++14 -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO -D_MWAITXINTRIN_H_INCLUDED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --source-in-ptx)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xptxas -dlcm=cg)

# enable NVCC warnings
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -Wall)

# the target architecture(s)  - (see https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_${CUDA_TARGET_COMPUTE_CAPABILITY},code=sm_${CUDA_TARGET_COMPUTE_CAPABILITY})
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_30,code=sm_30) # e.g. GTX 760
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_52,code=sm_52) # e.g. GTX 970
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_60,code=sm_60) # e.g. Tesla P100
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_61,code=sm_61) # e.g. GTX 1080Ti

# don't show deprecation warnings (which are a lot with CUDA 9)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-declarations)

set(CUDA_NVCC_FLAGS_DEBUG -g --generate-line-info --keep -DDEBUG)
set(CUDA_NVCC_FLAGS_RELEASE -O3 --generate-line-info --use_fast_math --keep -DNDEBUG)

set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION OFF)

include_directories(/usr/local/cuda/include)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -Og -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -march=corei7")

##############################################################################


# ---------------------------------------
# Add all sources to the variable SOURCES
# ---------------------------------------
#file(GLOB SOURCES "src/*.cpp")

# Add link_libraries
#add_library(student src/grade.cpp src/Student_info.cpp src/median.cpp)
#target_include_directories(student PUBLIC
#    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#    $<INSTALL_INTERFACE:include>
#    PRIVATE src)

# -----------------------
# Add sources separately
# -----------------------
SET(vec_src src/vecAdd.cu)
SET(matrix_src src/matrixMul.cu)
SET(vec_um_src src/vecAdd_UnifiedMemory.cu)
SET(saxpy_src src/saxpy.cu)
SET(test_src src/bandwidth_test.cu)
SET(coalescing_src src/coalescing.cu)
SET(reverse_src src/reverse_shared.cu)
SET(matrix_transpose_src src/matrix_transpose.cu)
SET(reduction src/reduction.cu)

# -----------------------
# Main target(s)
# -----------------------

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")

  #cuda_add_executable(vecAdd ${vec_src})
  #target_link_libraries(vecAdd cudart)

  #cuda_add_executable(matrixMul ${matrix_src})
  #target_link_libraries(matrixMul cudart)

  #cuda_add_executable(vecAdd_UnifiedMemory ${vec_um_src})
  #target_link_libraries(vecAdd_UnifiedMemory cudart)

  #cuda_add_executable(saxpy ${saxpy_src})
  #target_link_libraries(saxpy cudart)
  
  #cuda_add_executable(test ${test_src})
  #target_link_libraries(test cudart)
  
  #cuda_add_executable(coalescing ${coalescing_src})
  #target_link_libraries(coalescing cudart)

  #cuda_add_executable(reverse_shared ${reverse_src})
  #target_link_libraries(reverse_shared cudart)

  #cuda_add_executable(matrix_transpose ${matrix_transpose_src})
  #target_link_libraries(matrix_transpose cudart)
  
  cuda_add_executable(reduction ${reduction})
  target_link_libraries(reduction cudart)

endif()

# -------------

