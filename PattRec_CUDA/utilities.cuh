//
// Created by nicco on 03/03/20.
//

#ifndef PATTREC_UTILITIES_CUH
#define PATTREC_UTILITIES_CUH

#include <iostream>
#include <sstream>
#include <random>
#include <cstdlib>
#include <vector>
#include <list>
#include <chrono>
#include <algorithm>

/* Generate a sample historical data of specific length with a generator of rand number from 0 to 100. */
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

/* Generate a sample pattern data of specific length with a generator of rand number from 0 to 100. */
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

#endif // PATTREC_UTILITIES_CUH