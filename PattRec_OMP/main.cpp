//
// Created by nicco on 28/02/20.
//
#ifdef _OPENMP

#include <omp.h>

#endif

#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <experimental/filesystem>

#include "utilities.h"
#include "sequential.h"
#include "parallel.h"

namespace fs = std::experimental::filesystem;

int main(int argc, char *argv[]) {

    // take the current path and replace for correct saving path (Result/)
    // both using terminal or clion run.
    std::string path = fs::current_path();
    std::string r_path = "Result/";
    if (path.find("cmake") != std::string::npos) {
        path.replace(path.end() - 17, path.end(), r_path);
    } else { path += "/" + r_path; }

    printf("\nWelcome to Pattern Recognition in %s !!! \n\n", path.c_str());

    // define default hyper-parameters
    std::string path1 = "Result/";
    int NUM_QUERIES = 5;
    int LEN_SEQ = 10;
    int LEN_PATTERN_SEQ = 4;
    int verbose = 0;
    std::string type = "s";     // s for sequential mode or p for parallel query mode or p1 data level parallel
    std::string mode = "sequential";
    int iterations = 2;
    int RUNS = 2;
    int nthreads = 12;
    std::string par_data_type = "private";
    std::string testing_var = "nthreads"; // FIXME change the var if you change var for test!!!

    // set other hyper-parameters with launch arguments
    if (argc == 9) {
        // sequence length
        std::string s_LEN_SEQ = argv[1];
        std::stringstream parser1(s_LEN_SEQ);
        parser1 >> LEN_SEQ;

        // query length
        std::string s_LEN_PATTERN_SEQ = argv[2];
        std::stringstream parser2(s_LEN_PATTERN_SEQ);
        parser2 >> LEN_PATTERN_SEQ;

        // number of queries
        std::string s_NUM_QUERIES = argv[3];
        std::stringstream parser3(s_NUM_QUERIES);
        parser3 >> NUM_QUERIES;

        // total times in which there will be computed mean ad std
        std::string s_runs = argv[4];
        std::stringstream parser4(s_runs);
        parser4 >> RUNS;

        // times of iteration on the testing var
        std::string s_iter = argv[5];
        std::stringstream parser5(s_iter);
        parser5 >> iterations;

        // number of threads
        std::string s_nthr = argv[6];
        std::stringstream parser6(s_nthr);
        parser6 >> nthreads;

        // implementation type
        type = argv[7];

        std::string s_verbose = argv[8];
        std::stringstream parser8(s_verbose);
        parser8 >> verbose;

        if (LEN_SEQ < LEN_PATTERN_SEQ) {
            std::cout << "len of historical data less than len pattern seq!! Try again! " << std::endl;
            return 1;
        }

        std::cout << "You choose the following hyper-parameters: \n" << RUNS << " number of runs for mean and std; "
                  << type << " as type of execution\n" << NUM_QUERIES << " as number of queries; " << LEN_SEQ
                  << " as len of historical data;\n" << LEN_PATTERN_SEQ << " as len of each query; " << nthreads
                  << " as number of threads; " << verbose << " as verbose." << std::endl;
    }

    float *statistic;
    int size = iterations * 3;
    statistic = (float *) malloc(size * sizeof(float));

    for (int it = 0; it < iterations * 3; it = it + 3) {
        // FIXME change the var if you change var for test!!!
        printf("\n %s %d \n", testing_var.c_str(), LEN_SEQ);
        int LEN_RESULT = LEN_SEQ - LEN_PATTERN_SEQ + 1;

        // define path uniform distribution to sample data/query values
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
        std::vector<float> t_p2;


        for (int run = 0; run < RUNS; run++) {

            if (run % (RUNS / 2) == 0) std::cout << "STARTING RUN " << run << std::endl;

            float total_computational_time_seq = 0.0;
            float total_computational_time_par = 0.0;
            float total_computational_time_par2 = 0.0;
            float total_computational_time_par3 = 0.0;

            if (type == "s" or type == "all") {
                /* sequential execution */
                mode = "sequential";
                total_computational_time_seq = serialExecution(LEN_PATTERN_SEQ, LEN_RESULT, Historical_Data, Queries,
                                                               verbose);
                t_s.push_back(total_computational_time_seq);

                if (verbose > 0) {
                    std::cout << "\n\nFinal table \n    LEN SEQ: " << LEN_SEQ << "\n    LEN PATTERN SEQ: "
                              << LEN_PATTERN_SEQ << "\n    NUM QUERIES: " << NUM_QUERIES
                              << "\n    TOTAL COMPUTATION TIME: "
                              << total_computational_time_seq << " microsec" << "\n    EXECUTION: " << mode
                              << std::endl;
                }
            }

            if (type == "pq" or type == "all") {
                /* parallel execution on query*/
                mode = "parallel_lv_query";
                total_computational_time_par = parallelExecution_levQ(LEN_PATTERN_SEQ, LEN_RESULT, NUM_QUERIES,
                                                                      Historical_Data, Queries, nthreads, verbose);
                t_p.push_back(total_computational_time_par);

                if (verbose > 0) {
                    std::cout << "\n\nFinal table \n    LEN SEQ: " << LEN_SEQ << "\n    LEN PATTERN SEQ: "
                              << LEN_PATTERN_SEQ << "\n    NUM QUERIES: " << NUM_QUERIES
                              << "\n    TOTAL COMPUTATION TIME: "
                              << total_computational_time_par << " microsec" << "\n    EXECUTION: " << mode
                              << std::endl;
                }
            }

            if (type == "pd" or type == "all") {
                /* parallel execution on data (lock or privatization)*/
                mode = "parallel_lv_data";
                par_data_type = "private";
                total_computational_time_par2 = parallelExecution_levD(LEN_PATTERN_SEQ, LEN_RESULT, NUM_QUERIES,
                                                                       Historical_Data, Queries, nthreads, verbose,
                                                                       par_data_type);
                t_p1.push_back(total_computational_time_par2);

                if (verbose > 0) {
                    std::cout << "\n\nFinal table \n    LEN SEQ: " << LEN_SEQ << "\n    LEN PATTERN SEQ: "
                              << LEN_PATTERN_SEQ << "\n    NUM QUERIES: " << NUM_QUERIES
                              << "\n    TOTAL COMPUTATION TIME: "
                              << total_computational_time_par2 << " microsec" << "\n    EXECUTION: " << mode
                              << std::endl;
                }
            }

            if (type == "pdl" or type == "all") {
                /* parallel execution on data (lock or privatization)*/
                mode = "parallel_lv_data_with_lock";
                par_data_type = "lock";
                total_computational_time_par3 = parallelExecution_levD(LEN_PATTERN_SEQ, LEN_RESULT, NUM_QUERIES,
                                                                       Historical_Data, Queries, nthreads, verbose,
                                                                       par_data_type);
                t_p2.push_back(total_computational_time_par3);

                if (verbose > 0) {
                    std::cout << "\n\nFinal table \n    LEN SEQ: " << LEN_SEQ << "\n    LEN PATTERN SEQ: "
                              << LEN_PATTERN_SEQ << "\n    NUM QUERIES: " << NUM_QUERIES
                              << "\n    TOTAL COMPUTATION TIME: "
                              << total_computational_time_par3 << " microsec" << "\n    EXECUTION: " << mode
                              << std::endl;
                }
            }

            /*if (type != "s" and type != "pq" and verbose > 1) {
                float speed_up = total_computational_time_seq / total_computational_time_par;
                printf("\nSpeed Up: %f", speed_up);

                float speed_up2 = total_computational_time_seq / total_computational_time_par2;
                printf("\nSpeed Up: %f\n", speed_up2);
            }*/
        }

        // FIXME change the var if you change var for test!!!
        if (type == "s") {
            statistic[it] = compute_mean(t_s);
            statistic[it + 1] = compute_std(t_s);
            statistic[it + 2] = nthreads;
        }
        if (type == "pq") {
            statistic[it] = compute_mean(t_p);
            statistic[it + 1] = compute_std(t_p);
            statistic[it + 2] = nthreads;
        }
        if (type == "pd") {
            statistic[it] = compute_mean(t_p1);
            statistic[it + 1] = compute_std(t_p1);
            statistic[it + 2] = nthreads;
        }
        if (type == "pdl") {
            statistic[it] = compute_mean(t_p2);
            statistic[it + 1] = compute_std(t_p2);
            statistic[it + 2] = nthreads;
        }

        // FIXME change the var if you change var for test!!!
        // update len seq over iterations
        nthreads += 1;
        if (nthreads > 12) break;
    }

    save_result(statistic, size, mode, path, testing_var);

    return 0;
}
