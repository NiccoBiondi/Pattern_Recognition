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
    std::string path = "Result/";
    int NUM_QUERIES = 5;
    int LEN_SEQ = 10;
    int LEN_PATTERN_SEQ = 4;
    int verbose = 1;
    std::string type = "s";     // s for sequential mode or p for parallel query mode or p1 data level parallel
    std::string mode = "sequential";
    int iterations = 2;
    int RUNS = 2;

    // set other hyper-parameters with launch arguments
    if (argc == 8) {

        // lunghezza sequenza
        std::string s_LEN_SEQ = argv[1];
        std::stringstream parser1(s_LEN_SEQ);
        parser1 >> LEN_SEQ;

        // len single query
        std::string s_LEN_PATTERN_SEQ = argv[2];
        std::stringstream parser2(s_LEN_PATTERN_SEQ);
        parser2 >> LEN_PATTERN_SEQ;

        // numero totale di query
        std::string s_NUM_QUERIES = argv[3];
        std::stringstream parser3(s_NUM_QUERIES);
        parser3 >> NUM_QUERIES;

        // numero di volte in cui faccio media e std (10)
        std::string s_runs = argv[4];
        std::stringstream parser4(s_runs);
        parser4 >> RUNS;

        // numero di volte che voglio cambiare la lunghezza sequenza
        std::string s_iter = argv[5];
        std::stringstream parser5(s_iter);
        parser5 >> iterations;

        // tipologia technica
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

        std::vector<float> t_s;
        std::vector<float> t_p;
        std::vector<float> t_p1;

        for (int run = 0; run < RUNS; run++) {

            float total_computational_time_seq = 0.0;
            float total_computational_time_par = 0.0;
            float total_computational_time_par2 = 0.0;

            if (type == "s" or type == "a") {
                /* sequential execution */
                mode = "sequential";
                total_computational_time_seq = serialExecution(LEN_PATTERN_SEQ, LEN_RESULT, Historical_Data, Queries,
                                                               verbose);

                std::cout << "\n\nFinal table \n    LEN SEQ: " << LEN_SEQ << "\n    LEN PATTERN SEQ: "
                          << LEN_PATTERN_SEQ << "\n    NUM QUERIES: " << NUM_QUERIES << "\n    TOTAL COMPUTATION TIME: "
                          << total_computational_time_seq << " msec" << "\n    EXECUTION: " << mode << std::endl;
                t_s.push_back(total_computational_time_seq);
            }

            if (type == "p" or type == "a") {
                /* parallel execution */
                mode = "parallel_lv_query";
                total_computational_time_par = parallelExecution_levQ(LEN_PATTERN_SEQ, LEN_RESULT, NUM_QUERIES,
                                                                      Historical_Data, Queries, verbose);
                t_p.push_back(total_computational_time_par);
                std::cout << "\n\nFinal table \n    LEN SEQ: " << LEN_SEQ << "\n    LEN PATTERN SEQ: "
                          << LEN_PATTERN_SEQ << "\n    NUM QUERIES: " << NUM_QUERIES << "\n    TOTAL COMPUTATION TIME: "
                          << total_computational_time_par << " msec" << "\n    EXECUTION: " << mode << std::endl;
            }
            if (type == "p1" or type == "a") {
                total_computational_time_par2 = parallelExecution_levD(LEN_PATTERN_SEQ, LEN_RESULT, NUM_QUERIES,
                                                                       Historical_Data, Queries, verbose);
                t_p1.push_back(total_computational_time_par2);
                mode = "parallel_lv_data";
                std::cout << "\n\nFinal table \n    LEN SEQ: " << LEN_SEQ << "\n    LEN PATTERN SEQ: "
                          << LEN_PATTERN_SEQ << "\n    NUM QUERIES: " << NUM_QUERIES << "\n    TOTAL COMPUTATION TIME: "
                          << total_computational_time_par2 << " msec" << "\n    EXECUTION: " << mode << std::endl;
            }

            /*if (type != "s" and type != "p" and verbose > 1) {
                float speed_up = total_computational_time_seq / total_computational_time_par;
                printf("\nSpeed Up: %f", speed_up);

                float speed_up2 = total_computational_time_seq / total_computational_time_par2;
                printf("\nSpeed Up: %f\n", speed_up2);
            }*/
        }

        if (type == "s") {
            statistic[it] = compute_mean(t_s);
            statistic[it + 1] = compute_std(t_s);
            statistic[it + 2] = LEN_SEQ;
        }
        if (type == "p") {
            statistic[it] = compute_mean(t_p);
            statistic[it + 1] = compute_std(t_p);
            statistic[it + 2] = LEN_SEQ;
        }
        if (type == "p1") {
            statistic[it] = compute_mean(t_p1);
            statistic[it + 1] = compute_std(t_p1);
            statistic[it + 2] = LEN_SEQ;
        }

        // update len seq over iterations
        LEN_SEQ *= 5;
    }

    save_result(statistic, size, mode, path);

    return 0;
}
