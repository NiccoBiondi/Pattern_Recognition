//
// Created by nicco on 28/02/20.
//

#ifndef PARALLELCOMPUTING_SEQUENTIAL_H
#define PARALLELCOMPUTING_SEQUENTIAL_H

#include <iostream>
#include <sstream>
#include <random>
#include <cstdlib>
#include <vector>
#include <list>
#include <chrono>
#include <algorithm>

/* Define the sequential computation of SAD for one query on the historical data */
std::vector<int> queryExecution(int LEN_PATTERN_SEQ, int LEN_RESULT,
                                const std::vector<int> &data, const std::vector<int> &query, int verbose);

float serialExecution(int LEN_PATTERN_SEQ, int LEN_RESULT, const std::vector<int> &data,
                      const std::vector<std::vector<int>> &Queries, int verbose);

#endif //PARALLELCOMPUTING_SEQUENTIAL_H
