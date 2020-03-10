//
// Created by nicco on 04/03/20.
//

#include <iostream>
#include <chrono>
#include <vector>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <thrust/sort.h>

#include "execution.cuh"
#include "iteration.cuh"
#include "macros.h"

__constant__ float queries_const[MAX_LEN_Q];

int main(int argc, char **argv) {

#ifdef __CUDACC__
    std::cout << "cuda defined" << std::endl;
#endif

    std::cout << "Welcome to Pattern Recognition" << std::endl;

    // default hyper-parameters
    int LEN_SEQ = 10;
    int LEN_PATTERN_SEQ = 6;
    int NUM_QUERIES = 2;
    int verbose = 1;
    int iterations = 2;
    std::string type = "n";
    std::string mode = "tiling";     // mode: naive private tiling or constant
    // number of runs to compute speed up mean and std
    int RUNS = 2;

    // set other hyper-parameters with launch arguments
    if (argc == 8) {
        // convert the string argv[1] parameter in int
        std::string s_LEN_SEQ = argv[1];
        std::stringstream parser1(s_LEN_SEQ);
        parser1 >> LEN_SEQ;

        std::string s_LEN_PATTERN_SEQ = argv[2];
        std::stringstream parser2(s_LEN_PATTERN_SEQ);
        parser2 >> LEN_PATTERN_SEQ;

        std::string s_NUM_QUERIES = argv[3];
        std::stringstream parser3(s_NUM_QUERIES);
        parser3 >> NUM_QUERIES;

        std::string s_runs = argv[4];
        std::stringstream parser4(s_runs);
        parser4 >> RUNS;

        std::string s_iter = argv[5];
        std::stringstream parser5(s_iter);
        parser5 >> iterations;

        type = argv[6];

        std::string s_verbose = argv[7];
        std::stringstream parser6(s_verbose);
        parser6 >> verbose;

        if (LEN_SEQ < LEN_PATTERN_SEQ) {
            std::cout << "len of historical data less than len pattern seq!! Try again! " << std::endl;
            return 1;
        }

        std::cout << "You choose the following hyper-parameters: \n" << RUNS << " number of runs for mean and std; "
                  << type << " as type of execution\n " << NUM_QUERIES << " as number of queries; " << LEN_SEQ
                  << " as len of historical data; "
                  << LEN_PATTERN_SEQ << " as len of each query; " << verbose << " as verbose." << std::endl;
    }

    float *statistic;
    int size = iterations * 3;
    statistic = (float *) malloc(size * sizeof(float));

    for (int it = 0; it < iterations*3; it=it+3) {
        one_iteration(LEN_SEQ, LEN_PATTERN_SEQ, NUM_QUERIES, RUNS, type, mode, verbose, statistic, it);
        LEN_SEQ *= 5;
    }

    // save in csv statistics
    save_result(statistic, size, mode);

    return 0;
}