//
// Created by ncco on 03/03/20.
//

#ifndef PATTREC_EXECUTION_CUH
#define PATTREC_EXECUTION_CUH

#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define NUM_Q 2
#define LEN_QUERY 6
#define TILE_WIDTH 128
#define THREADS_PER_BLOCK TILE_WIDTH
#define CUDA_CHECK_RETURN(value) { gpuAssert((value), __FILE__, __LINE__); }
#define MAX_LEN_Q NUM_Q * LEN_QUERY

extern __constant__ float queries_const[MAX_LEN_Q];

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    /**
     * Check for errors in return values of CUDA functions
     **/
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void
computeSAD_naive(float *data, float *queries, float *result, int LEN_RESULT, int LEN_PATTERN_SEQ,
                 int NUM_QUERIES, float *minSad, int *minSadId);

__global__ void
computeSAD_priv(float *data, float *queries, float *result, int LEN_RESULT, int LEN_PATTERN_SEQ,
                int NUM_QUERIES, float *minSad, int *minSadId);


__global__ void
computeSAD_tiling(const float *data, const float *queries, float *result, int LEN_RESULT,
                  int LEN_PATTERN_SEQ, int NUM_QUERIES, float *minSad, int *minSadId);

__global__ void
computeSAD_constant(const float *data, float *result, int LEN_RESULT,
                    int LEN_PATTERN_SEQ, int NUM_QUERIES, float *minSad, int *minSadId);

#endif // PATTREC_EXECUTION_CUH