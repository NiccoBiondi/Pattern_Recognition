
cmake_minimum_required(VERSION 3.12)
project(Pattern_Recognition)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")

include_directories(PattRec_CUDA PattRec_OMP)

find_package(CUDA)

if (CUDA_FOUND)
    # using CUDA

    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=sm_61")
    enable_language(CUDA)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lcurand")
    add_executable(CUDA PattRec_CUDA/main.cu PattRec_CUDA/execution.cuh PattRec_CUDA/execution.cu)
    add_executable(OpenMP PattRec_OMP/main.cpp PattRec_OMP/parallel.h PattRec_OMP/parallel.cpp PattRec_OMP/sequential.h
            PattRec_OMP/sequential.cpp PattRec_OMP/utilities.h PattRec_OMP/utilities.cpp)
    target_link_libraries(CUDA -lcurand)
    set_target_properties(
            CUDA
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)


else ()
    # using CPU

    if (OPENMP_FOUND)
        # using OpenMP
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif (OPENMP_FOUND)

    add_executable(OpenMP PattRec_OMP/main.cpp PattRec_OMP/parallel.h PattRec_OMP/parallel.cpp PattRec_OMP/sequential.h
            PattRec_OMP/sequential.cpp PattRec_OMP/utilities.h PattRec_OMP/utilities.cpp)

endif (CUDA_FOUND)