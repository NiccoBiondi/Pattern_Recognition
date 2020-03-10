//
// Created by nicco on 09/03/20.
//

#include "iteration.cuh"

void one_iteration(int LEN_SEQ, int LEN_PATTERN_SEQ, int NUM_QUERIES, int RUNS, std::string type,
                   std::string mode, int verbose, float *statistic, int it) {

    /**
     * one iteration of the main loop. It takes some hyper-parameters and compute some (RUNS) runs to compute mean and
     * std of the some (type) modalities. Those values are stored in statistic for writing in csv file (see main).
     * **/

    std::cout << "\nThe new value of LEN SEQ is " << LEN_SEQ << std::endl;

    // compute hyper parameters after initialization
    int LEN_RESULT = LEN_SEQ - LEN_PATTERN_SEQ + 1;
    // check if the hyper pars are correct
    int gridX = ceil(LEN_SEQ / THREADS_PER_BLOCK) + 1;
    if (gridX < 1) {
        std::cout << "len seq is smaller than the THREADS_PER_BLOCK value!! Try again! " << std::endl;
        exit(1);
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

        if (run%(RUNS/2) == 0) std::cout << "STARTING RUN " << run << std::endl;

        // define data to store statistic of each run
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
            reset_result(result, result_ptr, LEN_RESULT, NUM_QUERIES);
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
            reset_result(result, result_ptr, LEN_RESULT, NUM_QUERIES);
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
            reset_result(result, result_ptr, LEN_RESULT, NUM_QUERIES);
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

        reset_result(result, result_ptr, LEN_RESULT, NUM_QUERIES);
    }

    std::cout << std::endl;
    // mean and std for each time vector
    if (type == "n" or type == "a") {
        double t_m_n = compute_mean(t_n);
        double std_n = compute_std(t_n);
        std::cout << "In " << RUNS << " runs the NAIVE mode reports " << t_m_n
                  << " microsec of mean with " << std_n << " of std" << std::endl;
        statistic[it] = t_m_n;
        statistic[it + 1] = std_n;
    }

    if (type == "p" or type == "a") {
        double t_m_p = compute_mean(t_p);
        double std_p = compute_std(t_p);
        std::cout << "In " << RUNS << " runs the PRIVATE mode reports " << t_m_p
                  << " microsec of mean with " << std_p << " of std" << std::endl;
        statistic[it] = t_m_p;
        statistic[it + 1] = std_p;
    }

    if (type == "t" or type == "a") {
        double t_m_t = compute_mean(t_t);
        double std_t = compute_std(t_t);
        std::cout << "In " << RUNS << " runs the TILING mode reports " << t_m_t
                  << " microsec of mean with " << std_t << " of std" << std::endl;
        statistic[it] = t_m_t;
        statistic[it + 1] = std_t;
    }

    if (type == "c" or type == "a") {
        double t_m_c = compute_mean(t_c);
        double std_c = compute_std(t_c);
        std::cout << "In " << RUNS << " runs the CONSTANT mode reports " << t_m_c
                  << " microsec of mean with " << std_c << " of std" << std::endl;
        statistic[it] = t_m_c;
        statistic[it + 1] = std_c;
    }

    // FIXME change the var if you change var for test!!!
    statistic[it + 2] = LEN_SEQ;

    // free host and device data
    curandDestroyGenerator(generator);
    free(data);
    cudaFree(data_ptr);

    free(queries);
    cudaFree(queries_ptr);

    free(minSad);
    free(minSadId);
    cudaFree(dev_minSad);
    cudaFree(dev_minSadId);

    cudaDeviceReset();

    return;
}