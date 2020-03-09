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

#include "execution.cuh"
#include "utilities.h"
#include "macros.h"

void one_iteration(int LEN_SEQ, int LEN_PATTERN_SEQ, int NUM_QUERIES, int RUNS, std::string type,
                   std::string mode, int verbose);

#endif //PATTERN_RECOGNITION_ITERATION_CUH
