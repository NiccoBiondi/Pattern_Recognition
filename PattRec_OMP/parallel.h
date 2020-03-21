//
// Created by nicco on 01/03/20.
//

#ifndef PARALLELCOMPUTING_PARALLEL_H
#define PARALLELCOMPUTING_PARALLEL_H

#ifdef _OPENMP

#include <omp.h>

#endif

#include <iostream>
#include <sstream>
#include <random>
#include <cstdlib>
#include <vector>
#include <list>
#include <chrono>
#include <algorithm>

/* Define the parallel computation of SAD for all the queries on the historical data with parallelism at queries level.
 * */
float parallelExecution_levQ(int LEN_PATTERN_SEQ, int LEN_RESULT, int NUM_QUERIES,
                             const std::vector<int> &data, const std::vector<std::vector<int>> &Queries,
                             int nthreads, int verbose);

/* Define the parallel computation of SAD for all the queries on the historical data with parallelism at data level.
 * */
float parallelExecution_levD(int LEN_PATTERN_SEQ, int LEN_RESULT, int NUM_QUERIES,
                             const std::vector<int> &data, const std::vector<std::vector<int>> &Queries,
                             int nthreads, int verbose, const std::string& par_data_type);

/* Define the parallel computation of SAD for one query on the historical data with result len nthread.
 * */
std::vector<int> queryParallelExecution(int LEN_PATTERN_SEQ, int LEN_RESULT, int nthreads,
                                        const std::vector<int> &data, const std::vector<int> &query);

/* Define the parallel computation of SAD for one query on the historical data with critical section.
 * */
std::vector<int> queryParallelExecution_lock(int LEN_PATTERN_SEQ, int LEN_RESULT, int nthreads,
                                             const std::vector<int> &data, const std::vector<int> &query);

/* Define the parallel computation of SAD for one query on the historical data without private thread variables.
 * */
/* FIXME: NON MI PIACE COME IMPLEMENTAZIONE SEMPRE LENTA CHE SI FA SI TIENE O SI CANCELLA??
 * std::vector<int> queryParallelExecution2(int LEN_PATTERN_SEQ, int LEN_RESULT,
                                        const std::vector<int> &data, const std::vector<int> &query, int verbose);*/

#endif //PARALLELCOMPUTING_PARALLEL_H
