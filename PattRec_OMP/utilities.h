//
// Created by nicco on 01/03/20.
//

#ifndef PARALLELCOMPUTING_UTILITIES_H
#define PARALLELCOMPUTING_UTILITIES_H

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
        std::uniform_int_distribution<int> &distribution, int verbose);

/* Generate a sample pattern data of specific length with a generator of rand number from 0 to 100. */
std::vector<int> generator_pattern(int LEN_PATTERN_SEQ, std::default_random_engine &generator,
        std::uniform_int_distribution<int> &distribution, int verbose);


#endif //PARALLELCOMPUTING_UTILITIES_H
