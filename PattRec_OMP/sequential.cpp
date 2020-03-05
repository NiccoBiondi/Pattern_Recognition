//
// Created by nicco on 28/02/20.
//

#include "sequential.h"

std::vector<int> queryExecution(int LEN_PATTERN_SEQ, int LEN_RESULT,
                                const std::vector<int> &data, const std::vector<int> &query, int verbose) {

    std::vector<int> result(LEN_RESULT);
    for (int i = 0; i < LEN_RESULT; i++) {
        for (int j = 0; j < LEN_PATTERN_SEQ; j++) {
            // compute SAD between data and query
            result[i] += abs(data[i + j] - query[j]);
        }
    }
    if (verbose != 0) printf("\nResult Vector of SAD: \n");
    for (int element : result) {
        if (verbose != 0) printf("%d ", element);
    }
    if (verbose != 0) printf("\n");

    // find the min SAD in result and its id
    auto min_SAD = std::min_element(result.begin(), result.end());
    int min_SAD_id = std::distance(result.begin(), min_SAD);

    std::vector<int> statistics = {result[min_SAD_id], min_SAD_id};

    return statistics;
}

float serialExecution(int LEN_PATTERN_SEQ, int LEN_RESULT, const std::vector<int> &data,
                      const std::vector<std::vector<int>> &Queries, int verbose) {

    float total_computational_time = 0;

    for (int i = 0; i < Queries.size(); i++) {
        // computational time for statistics of one query
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> statistics = queryExecution(LEN_PATTERN_SEQ, LEN_RESULT, data, Queries[i],
                                                     verbose);
        auto end = std::chrono::high_resolution_clock::now();

        auto computational_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        total_computational_time += computational_time;

        if (verbose != 0) printf("\nResulting statistic: %d, %d for query %d", statistics[0], statistics[1], i);

        if (verbose != 0) printf(" in    ");
        for (int j = statistics[1]; j < statistics[1] + LEN_PATTERN_SEQ; j++) {
            if (verbose != 0) printf("%d ", data[j]);
        }
        if (verbose != 0) printf("of the historical data \n\n");
    }

    return total_computational_time;
}