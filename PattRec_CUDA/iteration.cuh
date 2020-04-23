//
// Created by nicco on 09/03/20.
//

#ifndef PATTERN_RECOGNITION_ITERATION_CUH
#define PATTERN_RECOGNITION_ITERATION_CUH

#include <iostream>
#include <chrono>
#include <vector>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <thrust/sort.h>
#include <cstdlib>

#include "execution.cuh"
#include "utilities.h"
#include "macros.h"

std::vector<float *> allocate_queries(int LEN_PATTERN_SEQ, int NUM_QUERIES, std::string type, int verbose);

std::string one_iteration(int LEN_SEQ, int LEN_PATTERN_SEQ, int NUM_QUERIES, int RUNS, const std::string& type,
                   std::string mode, int verbose, float *statistic, int it);

inline void reset_result(float *result, float *result_ptr, int LEN_RESULT, int NUM_QUERIES) {
    /** clear the result both on host and device. After that free ptrs from memories. **/
    for (int i = 0; i < NUM_QUERIES * LEN_RESULT; i++) {
        result[i] = 0.0;
    }
    cudaMemcpy(result_ptr, result, NUM_QUERIES * LEN_RESULT * sizeof(float), cudaMemcpyHostToDevice);
    free(result);
    cudaFree(result_ptr);
}

inline void getMin(float *result, float *result_ptr, float *minSad, int *minSadId, int NUM_QUERIES, int LEN_PATTERN_SEQ,
                   int LEN_RESULT) {
    /**
     * find the min value in result vector sorting (with thrust) it and taking the first element (slow mode).
     * **/
    cudaMemcpy(result, result_ptr, NUM_QUERIES * LEN_RESULT * sizeof(float), cudaMemcpyDeviceToHost);
    for (int q = 0; q < NUM_QUERIES; q++) {
        thrust::sort(result + q * LEN_RESULT, result + (q + 1) * LEN_RESULT);
        minSad[q] = result[0 + q * LEN_RESULT];
        minSadId[q] = 0;
        printf("\n%f",minSad[q]);
    }
}

#endif //PATTERN_RECOGNITION_ITERATION_CUH
