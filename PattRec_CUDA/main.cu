//
// Created by nicco on 04/03/20.
//

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <thrust/sort.h>

#include "execution.cuh"

__constant__ float queries_const[MAX_LEN_Q];

double compute_mean(std::vector<float> v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double compute_std(std::vector<float> v) {
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    return std::sqrt(sq_sum / v.size());
}

int main(int argc, char **argv) {

#ifdef __CUDACC__
    std::cout << "cuda defined" << std::endl;
#endif

    std::cout << "Welcome to Pattern Recognition" << std::endl;

    // default hyper-parameters
    int LEN_SEQ = 10;
    int LEN_PATTERN_SEQ = 6;
    int NUM_QUERIES = 2;
    int verbose = 2;
    std::string type = "c";
    std::string mode = "naive";
    // number of runs to compute speed up mean and std
    int RUNS = 10;

    // set other hyper-parameters with launch arguments
    if (argc == 7) {
        // convert the string argv[1] parameter in int
        std::string s_LEN_SEQ = argv[1];
        std::stringstream parser1(s_LEN_SEQ);
        parser1 >> LEN_SEQ;

        std::string s_LEN_PATTERN_SEQ = argv[2];
        std::stringstream parser2(s_LEN_PATTERN_SEQ);
        parser2 >> LEN_PATTERN_SEQ;

        std::string s_NUM_QUERIES = argv[3];
        std::stringstream parser3(s_NUM_QUERIES);
        parser3 >> NUM_QUERIES;

        std::string s_runs = argv[4];
        std::stringstream parser4(s_runs);
        parser4 >> RUNS;

        type = argv[5];

        std::string s_verbose = argv[6];
        std::stringstream parser6(s_verbose);
        parser6 >> verbose;

        if (LEN_SEQ < LEN_PATTERN_SEQ) {
            std::cout << "len of historical data less than len pattern seq!! Try again! " << std::endl;
            return 1;
        }

        std::cout << "You choose the following hyper-parameters: \n" << RUNS << " number of runs for mean and std; "
                  << type << " as type of execution\n " << NUM_QUERIES << " as number of queries; " << LEN_SEQ
                  << " as len of historical data; "
                  << LEN_PATTERN_SEQ << " as len of each query; " << verbose << " as verbose." << std::endl;
    }

    // compute hyper parameters after initialization
    int LEN_RESULT = LEN_SEQ - LEN_PATTERN_SEQ + 1;
    // check if the hyper pars are correct
    int gridX = ceil(LEN_SEQ / THREADS_PER_BLOCK) + 1;
    if (gridX < 1) {
        std::cout << "len seq is smaller than the THREADS_PER_BLOCK value!! Try again! " << std::endl;
        return 1;
    }
    dim3 dimGrid(gridX, 1, 1);
    dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

    // define ptrs to data
    float *data, *queries;
    float *data_ptr, *queries_ptr;

    // allocate data on host and device
    data = (float *) malloc(LEN_SEQ * sizeof(float));
    queries = (float *) malloc(NUM_QUERIES * LEN_PATTERN_SEQ * sizeof(float));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &data_ptr, LEN_SEQ * sizeof(float)))
    CUDA_CHECK_RETURN(cudaMalloc((void **) &queries_ptr, NUM_QUERIES * LEN_PATTERN_SEQ * sizeof(float)))

    // generate data and queries on device
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
    curandGenerateUniform(generator, data_ptr, LEN_SEQ);
    curandGenerateUniform(generator, queries_ptr, NUM_QUERIES * LEN_PATTERN_SEQ);


    // the generator create data on device, if verbose != 0 you'll copy data on host for visualization
    if (verbose > 1) {
        cudaMemcpy(data, data_ptr, LEN_SEQ * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(queries, queries_ptr, NUM_QUERIES * LEN_PATTERN_SEQ * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "data : [";
        for (int i = 0; i < LEN_SEQ; i++) {
            std::cout << " " << data[i] << " ";
        }
        std::cout << "]" << std::endl;

        for (int q = 0; q < NUM_QUERIES * LEN_PATTERN_SEQ; q++) {
            if (q % LEN_PATTERN_SEQ == 0) std::cout << "query " << q / LEN_PATTERN_SEQ << ": [";
            std::cout << " " << queries[q] << " ";
            if (q % LEN_PATTERN_SEQ == (LEN_PATTERN_SEQ - 1)) std::cout << "]" << std::endl;
        }
    }

    // store the result
    float *minSad, *dev_minSad;
    int *minSadId, *dev_minSadId;
    minSad = (float *) malloc(NUM_QUERIES * sizeof(float));
    minSadId = (int *) malloc(NUM_QUERIES * sizeof(int));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &dev_minSad, NUM_QUERIES * sizeof(float)))
    CUDA_CHECK_RETURN(cudaMalloc((void **) &dev_minSadId, NUM_QUERIES * sizeof(int)))

    // vector for storing computational time
    std::vector<float> t_n;
    std::vector<float> t_p;
    std::vector<float> t_t;
    std::vector<float> t_c;
    float total_computational_time = 0.0;

    /***** Computing SAD on GPU *****/
    for (int run = 0; run < RUNS; run++) {

        std::cout << "\nSTARTING RUN " << run << std::endl;

        // define data to store results of each run
        float *result, *result_ptr;
        result = (float *) malloc(NUM_QUERIES * LEN_RESULT * sizeof(float));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &result_ptr, NUM_QUERIES * LEN_RESULT * sizeof(float)))

        if (type == "n" or type == "a") {
            mode = "naive";
            total_computational_time = 0.0;

            auto start = std::chrono::high_resolution_clock::now();
            computeSAD_naive<<<dimGrid, dimBlock>>>(data_ptr, queries_ptr, result_ptr, LEN_RESULT, LEN_PATTERN_SEQ,
                                                    NUM_QUERIES, dev_minSad, dev_minSadId);
            auto end = std::chrono::high_resolution_clock::now();
            total_computational_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            cudaMemcpy(minSad, dev_minSad, NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(minSadId, dev_minSadId, NUM_QUERIES * sizeof(int), cudaMemcpyDeviceToHost);

            if (verbose > 1) {
                cudaMemcpy(result, result_ptr, NUM_QUERIES * LEN_RESULT * sizeof(float), cudaMemcpyDeviceToHost);
                for (int r = 0; r < NUM_QUERIES * LEN_RESULT; r++) {
                    if (r % LEN_RESULT == 0) std::cout << "\nresult " << r / LEN_RESULT << ": [";
                    std::cout << " " << result[r] << " ";
                    if (r % LEN_RESULT == (LEN_RESULT - 1)) std::cout << "]" << std::endl;
                }
            }

            if (verbose >= 1) {
                std::cout << "\nMode " << mode << " in total computational time: " << total_computational_time
                          << " microsec" << std::endl;
                for (int s = 0; s < NUM_QUERIES; s++) {
                    std::cout << "Query " << s << " : min Sad = " << minSad[s] << " in Result ID = " << minSadId[s]
                              << std::endl;
                }
            }

            t_n.push_back(total_computational_time);
        }

        if (type == "a") {
            free(result);
            cudaFree(result_ptr);
            result = (float *) malloc(NUM_QUERIES * LEN_RESULT * sizeof(float));
            CUDA_CHECK_RETURN(cudaMalloc((void **) &result_ptr, NUM_QUERIES * LEN_RESULT * sizeof(float)))

        }

        if (type == "p" or type == "a") {
            mode = "private";
            total_computational_time = 0.0;

            auto start = std::chrono::high_resolution_clock::now();
            computeSAD_priv<<<dimGrid, dimBlock>>>(data_ptr, queries_ptr, result_ptr, LEN_RESULT, LEN_PATTERN_SEQ,
                                                   NUM_QUERIES, dev_minSad, dev_minSadId);
            auto end = std::chrono::high_resolution_clock::now();
            total_computational_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            cudaMemcpy(minSad, dev_minSad, NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(minSadId, dev_minSadId, NUM_QUERIES * sizeof(int), cudaMemcpyDeviceToHost);

            if (verbose > 1) {
                cudaMemcpy(result, result_ptr, NUM_QUERIES * LEN_RESULT * sizeof(float), cudaMemcpyDeviceToHost);
                for (int r = 0; r < NUM_QUERIES * LEN_RESULT; r++) {
                    if (r % LEN_RESULT == 0) std::cout << "\nresult " << r / LEN_RESULT << ": [";
                    std::cout << " " << result[r] << " ";
                    if (r % LEN_RESULT == (LEN_RESULT - 1)) std::cout << "]" << std::endl;
                }
            }

            if (verbose >= 1) {
                std::cout << "\nMode " << mode << " in total computational time: " << total_computational_time
                          << " microsec" << std::endl;
                for (int s = 0; s < NUM_QUERIES; s++) {
                    std::cout << "Query " << s << " : min Sad = " << minSad[s] << " in Result ID = " << minSadId[s]
                              << std::endl;
                }
            }

            t_p.push_back(total_computational_time);
        }

        if (type == "a") {
            free(result);
            cudaFree(result_ptr);

            result = (float *) malloc(NUM_QUERIES * LEN_RESULT * sizeof(float));
            CUDA_CHECK_RETURN(cudaMalloc((void **) &result_ptr, NUM_QUERIES * LEN_RESULT * sizeof(float)))

        }

        if (type == "t" or type == "a") {
            mode = "tiling";
            total_computational_time = 0.0;

            auto start = std::chrono::high_resolution_clock::now();
            computeSAD_tiling<<<dimGrid, dimBlock>>>(data_ptr, queries_ptr, result_ptr, LEN_RESULT, LEN_PATTERN_SEQ,
                                                     NUM_QUERIES, dev_minSad, dev_minSadId);
            auto end = std::chrono::high_resolution_clock::now();
            total_computational_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            cudaMemcpy(minSad, dev_minSad, NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(minSadId, dev_minSadId, NUM_QUERIES * sizeof(int), cudaMemcpyDeviceToHost);

            if (verbose > 1) {
                cudaMemcpy(result, result_ptr, NUM_QUERIES * LEN_RESULT * sizeof(float), cudaMemcpyDeviceToHost);
                for (int r = 0; r < NUM_QUERIES * LEN_RESULT; r++) {
                    if (r % LEN_RESULT == 0) std::cout << "\nresult " << r / LEN_RESULT << ": [";
                    std::cout << " " << result[r] << " ";
                    if (r % LEN_RESULT == (LEN_RESULT - 1)) std::cout << "]" << std::endl;
                }
            }

            if (verbose >= 1) {
                std::cout << "\nMode " << mode << " in total computational time: " << total_computational_time
                          << " microsec" << std::endl;
                for (int s = 0; s < NUM_QUERIES; s++) {
                    std::cout << "Query " << s << " : min Sad = " << minSad[s] << " in Result ID = " << minSadId[s]
                              << std::endl;
                }
            }

            t_t.push_back(total_computational_time);
        }

        if (type == "a") {
            free(result);
            cudaFree(result_ptr);
            result = (float *) malloc(NUM_QUERIES * LEN_RESULT * sizeof(float));
            CUDA_CHECK_RETURN(cudaMalloc((void **) &result_ptr, NUM_QUERIES * LEN_RESULT * sizeof(float)))
        }

        if (type == "c" or type == "a") {
            mode = "constant";
            total_computational_time = 0.0;

            // copy back the queries on host from device to use same values
            cudaMemcpy(queries, queries_ptr, NUM_QUERIES * LEN_PATTERN_SEQ * sizeof(float), cudaMemcpyDeviceToHost);
            // free device queries memory
            cudaFree(queries_ptr);
            // copy queries data on constant memory of device
            cudaMemcpyToSymbol(queries_const, queries, NUM_QUERIES * LEN_PATTERN_SEQ * sizeof(float));

            auto start = std::chrono::high_resolution_clock::now();
            computeSAD_constant<<<dimGrid, dimBlock>>>(data_ptr, result_ptr, LEN_RESULT, LEN_PATTERN_SEQ,
                                                       NUM_QUERIES, dev_minSad, dev_minSadId);
            auto end = std::chrono::high_resolution_clock::now();
            total_computational_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            cudaMemcpy(minSad, dev_minSad, NUM_QUERIES * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(minSadId, dev_minSadId, NUM_QUERIES * sizeof(int), cudaMemcpyDeviceToHost);

            if (verbose > 1) {
                cudaMemcpy(result, result_ptr, NUM_QUERIES * LEN_RESULT * sizeof(float), cudaMemcpyDeviceToHost);
                for (int r = 0; r < NUM_QUERIES * LEN_RESULT; r++) {
                    if (r % LEN_RESULT == 0) std::cout << "\nresult " << r / LEN_RESULT << ": [";
                    std::cout << " " << result[r] << " ";
                    if (r % LEN_RESULT == (LEN_RESULT - 1)) std::cout << "]" << std::endl;
                }
            }

            if (verbose >= 1) {
                std::cout << "\nMode " << mode << " in total computational time: " << total_computational_time
                          << " microsec" << std::endl;
                for (int s = 0; s < NUM_QUERIES; s++) {
                    std::cout << "Query " << s << " : min Sad = " << minSad[s] << " in Result ID = " << minSadId[s]
                              << std::endl;
                }
            }

            t_c.push_back(total_computational_time);
        }

        // delete result at the end of each run
        free(result);
        cudaFree(result_ptr);
    }

    std::cout << std::endl;
    // mean and std for each time vector
    if (type == "n" or type == "a") {
        double t_m_n = compute_mean(t_n);
        double std_n = compute_std(t_n);
        std::cout << "In " << RUNS << " runs the NAIVE mode reports " << t_m_n
                  << " microsec of mean with " << std_n << " of std" << std::endl;
    }

    if (type == "p" or type == "a") {
        double t_m_p = compute_mean(t_p);
        double std_p = compute_std(t_p);
        std::cout << "In " << RUNS << " runs the PRIVATE mode reports " << t_m_p
                  << " microsec of mean with " << std_p << " of std" << std::endl;
    }

    if (type == "t" or type == "a") {
        double t_m_t = compute_mean(t_t);
        double std_t = compute_std(t_t);
        std::cout << "In " << RUNS << " runs the TILING mode reports " << t_m_t
                  << " microsec of mean with " << std_t << " of std" << std::endl;
    }

    if (type == "c" or type == "a") {
        double t_m_c = compute_mean(t_c);
        double std_c = compute_std(t_c);
        std::cout << "In " << RUNS << " runs the CONSTANT mode reports " << t_m_c
                  << " microsec of mean with " << std_c << " of std" << std::endl;
    }

    // free host and device data
    curandDestroyGenerator(generator);
    free(data);
    free(queries);
    cudaFree(data_ptr);
    cudaFree(queries_ptr);

    free(minSad);
    free(minSadId);
    cudaFree(dev_minSad);
    cudaFree(dev_minSadId);

    cudaDeviceReset();

    return 0;
}