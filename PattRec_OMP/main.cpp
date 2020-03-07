//
// Created by nicco on 28/02/20.
//
#ifdef _OPENMP

#include <omp.h>

#endif

#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <algorithm>

#include "utilities.h"
#include "sequential.h"
#include "parallel.h"

int main(int argc, char *argv[]) {

    printf("\nWelcome to Pattern Recognition!!! \n\n");

    // define default hyper-parameters
    int NUM_QUERIES = 5;
    int LEN_SEQ = 10;
    int LEN_PATTERN_SEQ = 4;
    int verbose = 1;
    std::string type = "s";     // s for sequential mode  or p for parallel mode
    std::string mode = "sequential";

    // set other hyper-parameters with launch arguments
    if (argc == 6) {
        // convert the string argv[1] parameter in int
        std::string s_LEN_SEQ = argv[1];
        std::stringstream parser(s_LEN_SEQ);
        parser >> LEN_SEQ;

        std::string s_LEN_PATTERN_SEQ = argv[2];
        std::stringstream parser1(s_LEN_PATTERN_SEQ);
        parser1 >> LEN_PATTERN_SEQ;

        std::string s_NUM_QUERIES = argv[3];
        std::stringstream parser2(s_NUM_QUERIES);
        parser2 >> NUM_QUERIES;

        std::string s_verbose = argv[4];
        std::stringstream parser3(s_verbose);
        parser3 >> verbose;

        type = argv[5];
        if (type != "s" and type != "p") mode = "both parallel and sequential";

        if (LEN_SEQ < LEN_PATTERN_SEQ) {
            std::cout << "len of historical data less than len pattern seq!! Try again! " << std::endl;
            return 1;
        }

        std::cout << "You choose the following hyper-parameters: \n    " << mode << " execution; "
                  << NUM_QUERIES << " as number of queries; " << LEN_SEQ << " as len of historical data; "
                  << LEN_PATTERN_SEQ << " as len of each query; " << verbose << " as verbose." << std::endl;
    }


    int LEN_RESULT = LEN_SEQ - LEN_PATTERN_SEQ + 1;

    // define a uniform distribution to sample data/query values
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 100);

    // define list of vector of required data and iterators
    std::vector<int> Historical_Data = generator_data(LEN_SEQ, generator, distribution, verbose);
    std::vector<std::vector<int>> Queries(NUM_QUERIES);
    for (int i = 0; i < NUM_QUERIES; i++) {
        Queries[i] = generator_pattern(LEN_PATTERN_SEQ, generator, distribution, verbose);
    }

    float total_computational_time_seq = 0.0;
    float total_computational_time_par = 0.0;
    float total_computational_time_par2 = 0.0;

    if (type == "s" or type != "p") {
        /* sequential execution */
        mode = "sequential";
        total_computational_time_seq = serialExecution(LEN_PATTERN_SEQ, LEN_RESULT, Historical_Data, Queries, verbose);

        std::cout << "\n\nFinal table \n    LEN SEQ: " << LEN_SEQ << "\n    LEN PATTERN SEQ: "
                  << LEN_PATTERN_SEQ << "\n    NUM QUERIES: " << NUM_QUERIES << "\n    TOTAL COMPUTATION TIME: "
                  << total_computational_time_seq << " msec" << "\n    EXECUTION: " << mode << std::endl;

    }

    if (type == "p" or type != "s") {
        /* parallel execution */
        mode = "parallel level query";
        total_computational_time_par = parallelExecution_levQ(LEN_PATTERN_SEQ, LEN_RESULT, NUM_QUERIES,
                                                              Historical_Data, Queries, verbose);

        std::cout << "\n\nFinal table \n    LEN SEQ: " << LEN_SEQ << "\n    LEN PATTERN SEQ: "
                  << LEN_PATTERN_SEQ << "\n    NUM QUERIES: " << NUM_QUERIES << "\n    TOTAL COMPUTATION TIME: "
                  << total_computational_time_par << " msec" << "\n    EXECUTION: " << mode << std::endl;

        total_computational_time_par2 = parallelExecution_levD(LEN_PATTERN_SEQ, LEN_RESULT, NUM_QUERIES,
                                                               Historical_Data, Queries, verbose);

        mode = "parallel level data";
        std::cout << "\n\nFinal table \n    LEN SEQ: " << LEN_SEQ << "\n    LEN PATTERN SEQ: "
                  << LEN_PATTERN_SEQ << "\n    NUM QUERIES: " << NUM_QUERIES << "\n    TOTAL COMPUTATION TIME: "
                  << total_computational_time_par2 << " msec" << "\n    EXECUTION: " << mode << std::endl;
    }

    if (type != "s" and type != "p") {
        float speed_up = total_computational_time_seq / total_computational_time_par;
        printf("\nSpeed Up: %f", speed_up);

        float speed_up2 = total_computational_time_seq / total_computational_time_par2;
        printf("\nSpeed Up: %f\n", speed_up2);
    }

    return 0;
}
