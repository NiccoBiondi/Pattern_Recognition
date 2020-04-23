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
#include <experimental/filesystem>

#include "iteration.cuh"

__constant__ float queries_const[MAX_LEN_Q];
namespace fs = std::experimental::filesystem;

int main(int argc, char **argv) {

#ifdef __CUDACC__
    std::cout << "cuda defined" << std::endl;
#endif
    // take the current path and replace for correct saving path (Result/)
    // both using terminal or IDE run.
    std::string path = fs::current_path();
    std::string r_path = "test/Result/";
    printf("\nWelcome to Pattern Recognition in %s !!! \n\n", path.c_str());
    if (path.find("cmake") != std::string::npos){
        path.replace(path.end() - 17, path.end(), r_path);
    } else { path += "/" + r_path; }

    printf("\nWelcome to Pattern Recognition in %s !!! \n\n", path.c_str());

    // default hyper-parameters
    int LEN_SEQ = 10;
    int LEN_PATTERN_SEQ = 6;
    int NUM_QUERIES = 2;
    int verbose = 0;
    int iterations = 2;
    std::string type = "n";                       // type: n=naive, p=private, t=tiling, c=constant
    std::string mode = "naive";                   // mode: naive private tiling or constant
    int RUNS = 2;                                 // number of runs to compute computational time mean and std
    std::string testing_var = "LEN_SEQ";          // FIXME change the var if you change var for test!!!

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

    for (int it = 0; it < iterations * 3; it = it + 3) {
        // return mode that is the correct string for csv name
        mode = one_iteration(LEN_SEQ, LEN_PATTERN_SEQ, NUM_QUERIES, RUNS, type, mode, verbose, statistic, it);
        LEN_SEQ *= 2;   // FIXME change the var if you change var for test!!!
    }

    // save in csv statistics
    save_result(statistic, size, mode, path, testing_var);

    return 0;
}