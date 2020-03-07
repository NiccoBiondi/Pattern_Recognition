//
// Created by nicco on 04/03/20.
//

#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <sstream>

#define TILE_WIDTH 128
#define THREADS_PER_BLOCK TILE_WIDTH
#define CUDA_CHECK_RETURN(value) { gpuAssert((value), __FILE__, __LINE__); }

#include "execution.cuh"

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
    std::string type = "t";
    std::string mode = "naive";

    // set other hyper-parameters with launch arguments
    if (argc == 6) {
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

        std::string s_verbose = argv[4];
        std::stringstream parser4(s_verbose);
        parser4 >> verbose;

        type = argv[5];

        if (LEN_SEQ < LEN_PATTERN_SEQ) {
            std::cout << "len of historical data less than len pattern seq!! Try again! " << std::endl;
            return 1;
        }

        std::cout << "You choose the following hyper-parameters: \n    " << mode << " execution; "
                  << NUM_QUERIES << " as number of queries; " << LEN_SEQ << " as len of historical data; "
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
    float *data, *queries, *result;
    float *data_ptr, *queries_ptr, *result_ptr;
    // store the result
    float *minSad, *dev_minSad;
    int *minSadId, *dev_minSadId;

    // allocate data on host and device
    data = (float *) malloc(LEN_SEQ * sizeof(float));
    queries = (float *) malloc(NUM_QUERIES * LEN_PATTERN_SEQ * sizeof(float));
    result = (float *) malloc(NUM_QUERIES * LEN_RESULT * sizeof(float));

    CUDA_CHECK_RETURN(cudaMalloc((void **) &data_ptr, LEN_SEQ * sizeof(float)))
    CUDA_CHECK_RETURN(cudaMalloc((void **) &queries_ptr, NUM_QUERIES * LEN_PATTERN_SEQ * sizeof(float)))
    CUDA_CHECK_RETURN(cudaMalloc((void **) &result_ptr, NUM_QUERIES * LEN_RESULT * sizeof(float)))

    // allocate memory for storing the result
    minSad = (float *) malloc(NUM_QUERIES * sizeof(float));
    minSadId = (int *) malloc(NUM_QUERIES * sizeof(int));

    CUDA_CHECK_RETURN(cudaMalloc((void **) &dev_minSad, NUM_QUERIES * sizeof(float)))
    CUDA_CHECK_RETURN(cudaMalloc((void **) &dev_minSadId, NUM_QUERIES * sizeof(int)))

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

    float total_computational_time = 0.0;

    /** Computing SAD on GPU **/
    if (type == "n" or type == "a") {
        mode = "naive";
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

        std::cout << "\nMode " << mode << " in total computational time: " << total_computational_time << " microsec"
                  << std::endl;
        if (verbose >= 1) {
            for (int s = 0; s < NUM_QUERIES; s++) {
                std::cout << "Query " << s << " : min Sad = " << minSad[s] << " in Result ID = " << minSadId[s]
                          << std::endl;
            }
        }
    }

    if (type == "p" or type == "a") {
        mode = "private";
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

        std::cout << "\nMode " << mode << " in total computational time: " << total_computational_time << " microsec"
                  << std::endl;
        if (verbose >= 1) {
            for (int s = 0; s < NUM_QUERIES; s++) {
                std::cout << "Query " << s << " : min Sad = " << minSad[s] << " in Result ID = " << minSadId[s]
                          << std::endl;
            }
        }
    }

    if (type == "t" or type == "a") {
        mode = "tiling";
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

        std::cout << "\nMode " << mode << " in total computational time: " << total_computational_time << " microsec"
                  << std::endl;
        if (verbose >= 1) {
            for (int s = 0; s < NUM_QUERIES; s++) {
                std::cout << "Query " << s << " : min Sad = " << minSad[s] << " in Result ID = " << minSadId[s]
                          << std::endl;
            }
        }
    }

    if (type == "c" or type == "a") {
        // FIXME finish me
        mode = "constant";
        auto start = std::chrono::high_resolution_clock::now();
        computeSAD_constant<<<dimGrid, dimBlock>>>(data_ptr, queries_ptr, result_ptr, LEN_RESULT, LEN_PATTERN_SEQ,
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

        std::cout << "\nMode " << mode << " in total computational time: " << total_computational_time << " microsec"
                  << std::endl;
        if (verbose >= 1) {
            for (int s = 0; s < NUM_QUERIES; s++) {
                std::cout << "Query " << s << " : min Sad = " << minSad[s] << " in Result ID = " << minSadId[s]
                          << std::endl;
            }
        }
    }


    // free host and device data
    curandDestroyGenerator(generator);
    free(data);
    free(queries);
    free(result);
    cudaFree(data_ptr);
    cudaFree(queries_ptr);
    cudaFree(result_ptr);

    free(minSad);
    free(minSadId);
    cudaFree(dev_minSad);
    cudaFree(dev_minSadId);

    return 0;
}