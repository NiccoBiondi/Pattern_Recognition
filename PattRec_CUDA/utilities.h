//
// Created by nicco on 09/03/20.
//

#ifndef PATTERN_RECOGNITION_UTILITIES_H
#define PATTERN_RECOGNITION_UTILITIES_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "macros.h"

inline double compute_mean(std::vector<float> v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

inline double compute_std(std::vector<float> v) {
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    return std::sqrt(sq_sum / v.size());
}

inline void save_result(const std::vector<float> &v, const std::string& mode) {
    std::string path = "/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/" + mode + ".csv";
    std::ofstream csvFile(path);
    for (int r = 0; r < v.size(); r++) {
        csvFile << v[r] << std::endl;
    }
    csvFile.close();
}

void one_iteration();

#endif //PATTERN_RECOGNITION_UTILITIES_H
