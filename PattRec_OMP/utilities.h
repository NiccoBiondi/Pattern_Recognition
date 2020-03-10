//
// Created by nicco on 01/03/20.
//

#ifndef PARALLELCOMPUTING_UTILITIES_H
#define PARALLELCOMPUTING_UTILITIES_H

#include <iostream>
#include <fstream>
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

inline void
save_result(const float *v, int size, const std::string &mode, std::string save_path, std::string testing_var) {
    std::string path = save_path + "CPP_" + mode + "_" + testing_var + ".csv";
    std::ofstream csvFile(path);
    for (int r = 0; r < size; r = r + 3) {
        // write mean,std
        csvFile << v[r] << ",";
        csvFile << v[r + 1] << ",";
        csvFile << v[r + 2] << std::endl;
    }
    csvFile.close();
}

inline float compute_mean(std::vector<float> v) {
    return (float) std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

inline float compute_std(std::vector<float> v) {
    float mean = (float) std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    std::vector<float> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mean](float x) { return x - mean; });
    float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    return (float) std::sqrt(sq_sum / v.size());
}

#endif //PARALLELCOMPUTING_UTILITIES_H
