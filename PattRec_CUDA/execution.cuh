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

__global__ void
computeSAD_naive(float *data, float *queries, float *result, int LEN_RESULT, int LEN_PATTERN_SEQ,
                 int NUM_QUERIES, float *minSad, int *minSadId) {
    /**
     * Compute result array reading both queries and data from global memory ?with less mem access? CHECK ME
     * (aka naive implementation)
     **/

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    //int LEN_SEQ = LEN_PATTERN_SEQ + LEN_RESULT - 1;

    //if ((index + LEN_PATTERN_SEQ - 1 < LEN_SEQ) && (threadIdx.x < LEN_RESULT)) {
    if (index < LEN_RESULT) {
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
    float t_SAD = 0;

    for (int q = 0; q < NUM_QUERIES; q++) {
        t_SAD = 0;
        //if ((index + LEN_PATTERN_SEQ - 1 < LEN_SEQ) && (threadIdx.x < LEN_RESULT)) {
        if (index < LEN_RESULT) {
            for (int i = 0; i < LEN_PATTERN_SEQ; i++) {
                t_SAD += abs(data[index + i] - queries[i + q * LEN_PATTERN_SEQ]);
            }
            __syncthreads();
            result[index + q * LEN_RESULT] = t_SAD;
        }
    }

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
computeSAD_tiling(const float *data, const float *queries, float *result, int LEN_RESULT,
                  int LEN_PATTERN_SEQ, int NUM_QUERIES, float *minSad, int *minSadId) {

    /**
     * Compute result array reading data and query from shared mem (aka tiling implementation)
     **/

    __shared__ float data_sh[TILE_WIDTH];
    __shared__ float query_sh[TILE_WIDTH];

    int LEN_SEQ = LEN_PATTERN_SEQ + LEN_RESULT - 1;
    int index = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (index < LEN_SEQ) {
        data_sh[threadIdx.x] = data[index];
    } else { data_sh[threadIdx.x] = 0.0; }
    __syncthreads();    // tiled data loaded on shared mem of the block

    for (int q = 0; q < NUM_QUERIES; q++) {
        for (int p = 0; p < (LEN_PATTERN_SEQ - 1) / TILE_WIDTH + 1; p++) {
            if ((threadIdx.x + p * TILE_WIDTH) < LEN_PATTERN_SEQ) {
                query_sh[threadIdx.x] = queries[threadIdx.x + p * TILE_WIDTH + q * LEN_PATTERN_SEQ];
            } else { query_sh[threadIdx.x] = 0.0; }
            __syncthreads();    // tiled query loaded on shared mem of the block

            if (index < LEN_SEQ) {
                for (int r = 0; r < TILE_WIDTH; r++) {
                    if (0 <= (index - (r + p * TILE_WIDTH)) and (index - (r + p * TILE_WIDTH)) < LEN_RESULT and
                        query_sh[r] != 0.0) {
                        atomicAdd(&(result[(index - (r + p * TILE_WIDTH)) + q * LEN_RESULT]), abs(
                                data_sh[threadIdx.x] - query_sh[r]));
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }

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
computeSAD_constant(const float *data, const float *queries, float *result, int LEN_RESULT,
                    int LEN_PATTERN_SEQ, int NUM_QUERIES, float *minSad, int *minSadId) {
    /**
     * Compute SAD loading the queries in the constant memory and tiling the data (shared mem)
     * (aka constant implementation)
     **/

    __shared__ float data_sh[TILE_WIDTH];

    int LEN_SEQ = LEN_PATTERN_SEQ + LEN_RESULT - 1;
    int index = threadIdx.x + TILE_WIDTH * blockIdx.x;

    if (index < LEN_SEQ) {
        data_sh[threadIdx.x] = data[index];
    } else { data_sh[threadIdx.x] = 0.0; }
    __syncthreads();    // tiled data loaded on shared mem of the block

    if (index < LEN_SEQ) {
        for (int q = 0; q < NUM_QUERIES; q++) {
            for (int r = 0; r < LEN_PATTERN_SEQ; r++) {
                if (0 <= (index - r) and (index - r) < LEN_RESULT and queries[r] != 0.0) {
                    atomicAdd(&(result[(index - r) + q * LEN_RESULT]), abs(
                            data_sh[threadIdx.x] - queries[r + q * LEN_PATTERN_SEQ]));
                }
            }
        }
    }

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

#endif // PATTREC_EXECUTION_CUH