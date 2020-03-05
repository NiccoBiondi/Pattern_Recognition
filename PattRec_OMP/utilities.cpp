//
// Created by nicco on 01/03/20.
//

#include "utilities.h"

std::vector<int> generator_data(int LEN_SEQ, std::default_random_engine &generator,
        std::uniform_int_distribution<int> &distribution, int verbose) {

    std::vector<int> historical_data(LEN_SEQ);

    if (verbose != 0) printf("Historical Data: \n");
    for (int i = 0; i < LEN_SEQ; i++) {
        historical_data[i] = distribution(generator);
        if (verbose != 0) printf("%d ", historical_data[i]);
    }
    if (verbose != 0) printf("\n");

    return historical_data;
}


std::vector<int> generator_pattern(int LEN_PATTERN_SEQ, std::default_random_engine &generator,
        std::uniform_int_distribution<int> &distribution, int verbose) {

    std::vector<int> query(LEN_PATTERN_SEQ);

    if (verbose != 0) printf("Query: \n");
    for (int i = 0; i < LEN_PATTERN_SEQ; i++) {
        query[i] = distribution(generator);
        if (verbose != 0) printf("%d ", query[i]);
    }
    if (verbose != 0) printf("\n");

    return query;
}

