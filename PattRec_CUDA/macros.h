//
// Created by nicco on 09/03/20.
//

#ifndef PATTERN_RECOGNITION_MACROS_H
#define PATTERN_RECOGNITION_MACROS_H

#define NUM_Q 10
#define LEN_QUERY 1000
#define TILE_WIDTH 64
#define THREADS_PER_BLOCK TILE_WIDTH
#define CUDA_CHECK_RETURN(value) { gpuAssert((value), __FILE__, __LINE__); }
#define MAX_LEN_Q NUM_Q * LEN_QUERY

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    /**
     * Check for errors in return values of CUDA functions
     **/
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif //PATTERN_RECOGNITION_MACROS_H
