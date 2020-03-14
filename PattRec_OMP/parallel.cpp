//
// Created by nicco on 01/03/20.
//
#include "parallel.h"
#include "sequential.h"

float parallelExecution_levQ(int LEN_PATTERN_SEQ, int LEN_RESULT, int NUM_QUERIES,
                             const std::vector<int> &data, const std::vector<std::vector<int>> &Queries,
                             int verbose) {

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel default(none) shared(LEN_PATTERN_SEQ, LEN_RESULT, NUM_QUERIES, data, Queries, verbose)
    {
#pragma omp for
        for (int i = 0; i < NUM_QUERIES; i++) {
            std::vector<int> query_statistics = queryExecution(LEN_PATTERN_SEQ, LEN_RESULT, data, Queries[i],
                                                               verbose);
            if (verbose != 0)
                printf("\nResulting statistic: %d, %d for query %d", query_statistics[0], query_statistics[1], i);
        }
    }


    auto end = std::chrono::high_resolution_clock::now();
    float total_computational_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return total_computational_time;
}

float parallelExecution_levD(int LEN_PATTERN_SEQ, int LEN_RESULT, int NUM_QUERIES,
                             const std::vector<int> &data, const std::vector<std::vector<int>> &Queries,
                             int verbose) {

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_QUERIES; i++) {
        std::vector<int> query_statistics = queryParallelExecution(LEN_PATTERN_SEQ, LEN_RESULT, data, Queries[i]);
        if (verbose != 0)
            printf("\nResulting statistic: %d, %d for query %d", query_statistics[0], query_statistics[1], i);
    }

    auto end = std::chrono::high_resolution_clock::now();
    float total_computational_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return total_computational_time;
}

std::vector<int> queryParallelExecution(int LEN_PATTERN_SEQ, int LEN_RESULT, const std::vector<int> &data,
                                        const std::vector<int> &query) {

    int nthreads = omp_get_num_procs();
    int SAD, tid;
    int thread_min = LEN_PATTERN_SEQ * 100;
    std::vector<int> result(nthreads);

#pragma omp parallel default(none) firstprivate(SAD, thread_min, tid) shared(data, query, LEN_RESULT, LEN_PATTERN_SEQ, result)
    {
        tid = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < LEN_RESULT; i++) {
            SAD = 0;
            for (int j = 0; j < LEN_PATTERN_SEQ; j++) {
                SAD += abs(data[i + j] - query[j]);
            }
            if (SAD < thread_min) {
                thread_min = SAD;
            }
        }
        result[tid] = thread_min;
    };

    // find the min SAD in result and its id
    auto min_SAD = std::min_element(result.begin(), result.end());
    int min_SAD_id = std::distance(result.begin(), min_SAD);

    std::vector<int> statistics = {result[min_SAD_id], min_SAD_id};

    return statistics;
}

std::vector<int> queryParallelExecution_lock(int LEN_PATTERN_SEQ, int LEN_RESULT, const std::vector<int> &data,
                                             const std::vector<int> &query) {

    int SAD, tid;
    int min_SAD_id = 0;
    int thread_min = LEN_PATTERN_SEQ * 100;
    int min_SAD = LEN_PATTERN_SEQ * 100;

#pragma omp parallel default(none) firstprivate(SAD, thread_min, tid) shared(data, query, LEN_RESULT, LEN_PATTERN_SEQ, min_SAD, min_SAD_id)
    {
        tid = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < LEN_RESULT; i++) {
            SAD = 0;
            for (int j = 0; j < LEN_PATTERN_SEQ; j++) {
                SAD += abs(data[i + j] - query[j]);
            }
            if (SAD < thread_min) {
                thread_min = SAD;
            }
        } // each thread compute its min SAD
#pragma omp flush(min_SAD)
        // flush global min and eventually update it
        if (thread_min < min_SAD) {
#pragma omp critical
            if (thread_min < min_SAD) {
                min_SAD = thread_min;
                min_SAD_id = tid;
            }
        }
    }

    std::vector<int> statistics = {min_SAD, min_SAD_id};
    return statistics;
}
