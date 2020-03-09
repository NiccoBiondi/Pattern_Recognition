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

#include "macros.h"

extern __constant__ float queries_const[MAX_LEN_Q];

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