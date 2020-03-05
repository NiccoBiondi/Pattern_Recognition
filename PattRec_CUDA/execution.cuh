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

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    /**
     * Check for errors in return values of CUDA functions
     **/
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// FIXME INUTILE
__global__ void hellGPU(int LEN_SEQ, int LEN_PATTERN_SEQ) {

    int my_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x % 2) {
        printf("\n %d %d %d %d", LEN_SEQ, blockIdx.x, blockDim.x, threadIdx.x);
    } else if (blockIdx.x % 3) {
        printf("\n qua qua si fa qua qua ");
    }

    // some threads of the blocks will be not used check for them...
}
// FIXME INUTILE
__global__ void add(const int *a, const int *b, int *c) {

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

// FIXME INUTILE
__global__ void dot_prod(const int *a, const int *b, int *c) {
    __shared__ int temp[THREADS_PER_BLOCK];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = a[index] * b[index];

    __syncthreads();

    if (0 == threadIdx.x) {
        int sum = 0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) {
            sum += temp[i];
        }
        atomicAdd(c, sum);
    }
}

__global__ void
computeSAD_naive(float *data, float *queries, float *result, int LEN_RESULT, int LEN_PATTERN_SEQ,
                 int NUM_QUERIES, float *minSad, int *minSadId) {
    /**
     * Compute result array reading both queries and data from global memory ?with less mem access? CHECK ME
     * (aka naive implementation)
     **/

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int LEN_SEQ = LEN_PATTERN_SEQ + LEN_RESULT - 1;

    if ((index + LEN_PATTERN_SEQ - 1 < LEN_SEQ) && (threadIdx.x < LEN_RESULT)) {
        for (int i = 0; i < LEN_PATTERN_SEQ; i++) {
            for (int q = 0; q < NUM_QUERIES; q++) {
                result[index + q * LEN_RESULT] += abs(data[index + i] - queries[i + q * LEN_PATTERN_SEQ]);
            }
        }
    }

    // Barrier for threads in a block
    __syncthreads();

    // one thread search for each result vector the min SAD and the ID
    for (int q = 0; q < NUM_QUERIES; q++) {
        if (index == q) {
            auto min = (float) LEN_PATTERN_SEQ; // max value since every entry is <= 1
            int min_index;
            for (int i = 0; i < LEN_RESULT; i++) {
                if (result[i + q * LEN_RESULT] < min) {
                    min = result[i + q * LEN_RESULT];
                    min_index = i;
                }
            }
            minSad[q] = min;
            minSadId[q] = min_index;
        }
    }
}

__global__ void
computeSAD_priv(float *data, float *queries, float *result, int LEN_RESULT, int LEN_PATTERN_SEQ,
                int NUM_QUERIES, float *minSad, int *minSadId) {

    /**
     * Compute result array reading both queries and data from global memory with privatization
     * (aka tmpMin implementation)
     **/

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int LEN_SEQ = LEN_PATTERN_SEQ + LEN_RESULT - 1;
    float t_min;

    for (int q = 0; q < NUM_QUERIES; q++) {
        t_min = 0;
        if ((index + LEN_PATTERN_SEQ - 1 < LEN_SEQ) && (threadIdx.x < LEN_RESULT)) {
            for (int i = 0; i < LEN_PATTERN_SEQ; i++) {
                t_min += abs(data[index + i] - queries[i + q * LEN_PATTERN_SEQ]);
            }
            __syncthreads();
            result[index + q * LEN_RESULT] = t_min;
        }
    }
}


__global__ void
computeSAD_tiling(const float *data, const float *queries, float *result, int LEN_RESULT,
                  int LEN_PATTERN_SEQ, int NUM_QUERIES, float *minSad, int *minSadId) {

    /**
     * Compute result array reading data and query from shared mem (aka tiling implementation)
     **/

    __shared__ float data_sh[TILE_WIDTH];
    __shared__ float query_sh[TILE_WIDTH];

    int LEN_SEQ = LEN_PATTERN_SEQ + LEN_RESULT - 1;
    int index = threadIdx.x + blockIdx.x * TILE_WIDTH;
    float tmp_sad;
    //printf("\n%d, %d, %d", threadIdx.x, blockIdx.x, index);

    for (int q = 0; q < NUM_QUERIES; q++) {
        for (int p = 0; p < (LEN_SEQ - 1) / TILE_WIDTH + 1; ++p) {
            if (p * TILE_WIDTH + threadIdx.x < LEN_RESULT) {
                for (int i = 0; i < LEN_PATTERN_SEQ; i++) {
                    // we also load with it its next values
                    data_sh[threadIdx.x + i] = data[p * TILE_WIDTH + threadIdx.x + i];
                    query_sh[i] = queries[i + q * LEN_RESULT];
                }
            } else { data_sh[threadIdx.x] = 0.0; }
            __syncthreads();
            if (index < LEN_RESULT) {
                for (int i = 0; i < TILE_WIDTH; ++i) {
                    tmp_sad += abs(data_sh[i] - query_sh[i + q * LEN_PATTERN_SEQ]);
                }
            }
            __syncthreads();
            if (index < LEN_RESULT) {
                result[index + q * LEN_RESULT] = tmp_sad;
            }
        }
    }


    /*
    for (int q = 0; q < NUM_QUERIES; q++) {
        for (int i = 0; i < LEN_SEQ / TILE_WIDTH; i++) {
            // check if in tile can enter all the sequence for compute SAD
            if (threadIdx.x + LEN_PATTERN_SEQ - 1 < TILE_WIDTH) {
                for (int j = 0; j < LEN_PATTERN_SEQ; j++) {
                    data_sh[threadIdx.x + j + i * TILE_WIDTH] = data[index + j + i * TILE_WIDTH];
                    query_sh[threadIdx.x + j + i * TILE_WIDTH + q * LEN_PATTERN_SEQ] = queries[threadIdx.x +
                                                                                               q * LEN_PATTERN_SEQ + j +
                                                                                               i * TILE_WIDTH];
                    printf("\ndata %f", data_sh[threadIdx.x + j + i * TILE_WIDTH]);
                }
            }
            // Barrier for threads in a block
            __syncthreads();
            for (int t = 0; t < TILE_WIDTH; t++) {
                tmp += abs(data_sh[t] - query_sh[t + q * LEN_PATTERN_SEQ]);
            }
            result[index + q * LEN_RESULT] = tmp;
        }
    }*/

    /*float tmp_sad = 0;

    for (int i = 0; i < LEN_PATTERN_SEQ / TILE_WIDTH; ++i) {
        // Load into shared mem
        data_sh[tile_id] = data[i * TILE_WIDTH + threadIdx.x];
        printf("\n%f ", data_sh[tile_id]);
        queries_sh[tile_id] = query[i * TILE_WIDTH + threadIdx.x];
        // Barrier for threads in a block
        __syncthreads();
    }

    if (index < LEN_RESULT) {
        for (int i = 0; i < LEN_PATTERN_SEQ; i++) {
            result[index] += abs(data[index + i] - query[i]);
        }
    }

    // Barrier for threads in a block
    __syncthreads();

    if (0 == threadIdx.x) {
        auto min = (float) LEN_PATTERN_SEQ; // max value
        int min_index;
        for (int i = 0; i < LEN_RESULT; i++) {
            if (result[i] < min) {
                min = result[i];
                min_index = i;
            }
        }
        *minSad = min;
        *minSadId = min_index;
    }*/
}

#endif // PATTREC_EXECUTION_CUH