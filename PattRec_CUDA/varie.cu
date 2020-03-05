//
//  Created by nicco on 03/03/20.
//
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>


#include "utilities.cuh"
#include "execution.cuh"

void random_ints(int* x, int size)
{
    int i;
    for (i=0;i<size;i++) {
        x[i]=rand()%10;
    }
}


int main(int argc, char **argv) {
#ifdef __CUDACC__
    std::cout << "cuda cc enabled" << std::endl;
#endif

    std::cout << "Welcome to Pattern Recognition!" << std::endl;

    // define default hyper-parameters
    int NUM_QUERIES = 5;
    int LEN_SEQ = 100000000;
    int LEN_PATTERN_SEQ = 10;
    int verbose = 0;
    std::string type = "s";     // s for sequential mode  or p for parallel mode
    std::string mode = "sequential";

    // set other hyper-parameters with launch arguments
    if (argc == 6) {
        // convert the string argv[1] parameter in int
        std::string s_NUM_QUERIES = argv[1];
        std::stringstream parser(s_NUM_QUERIES);
        parser >> NUM_QUERIES;

        std::string s_LEN_SEQ = argv[2];
        std::stringstream parser1(s_LEN_SEQ);
        parser1 >> LEN_SEQ;

        std::string s_LEN_PATTERN_SEQ = argv[3];
        std::stringstream parser2(s_LEN_PATTERN_SEQ);
        parser2 >> LEN_PATTERN_SEQ;

        std::string s_verbose = argv[4];
        std::stringstream parser3(s_verbose);
        parser3 >> verbose;

        type = argv[5];
        if (type != "s" and type != "p") mode = "both parallel and sequential";

        if (LEN_SEQ < LEN_PATTERN_SEQ) {
            std::cout << "len of historical data less than len pattern seq!! Try again! " << std::endl;
            return 1;
        }

        std::cout << "You choose the following hyper-parameters: \n    " << mode << " execution; "
                  << NUM_QUERIES << " as number of queries; " << LEN_SEQ << " as len of historical data; "
                  << LEN_PATTERN_SEQ << " as len of each query; " << verbose << " as verbose." << std::endl;
    }


    int LEN_RESULT = LEN_SEQ - LEN_PATTERN_SEQ;

    // define a uniform distribution to sample data/query values
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 100);

    // define list of vector of required data and iterators
    std::vector<int> Historical_Data = generator_data(LEN_SEQ, generator, distribution, verbose);
    std::vector<std::vector<int>> Queries(NUM_QUERIES);

    for (int i = 0; i < NUM_QUERIES; i++) {
        Queries[i] = generator_pattern(LEN_PATTERN_SEQ, generator, distribution, verbose);
    }

    dim3 dimGrid(ceil(Historical_Data.size() / 128.0), 1, 1);           // number of blocks (aka grid dim)
    dim3 dimBlock(128, 1, 1);                                              // number of thread per block (aka block dim)
    std::cout << "\nqueste sono qua " << dimGrid.x << " " << dimBlock.x << std::endl;

    // hellGPU<<<dimGrid, dimBlock>>>(LEN_SEQ, LEN_PATTERN_SEQ, Historical_Data);

    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;
    int size = N * sizeof(int);

    // allocate memories in device
    cudaMalloc((void **) &dev_a, size);
    cudaMalloc((void **) &dev_b, size);
    cudaMalloc((void **) &dev_c, sizeof(int));

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(sizeof(int));

    random_ints(a,N);
    random_ints(b,N);

    std::cout << "A[";
    for(int i=0; i<N; i++) {
        std::cout << " " << a[i] << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "B[";
    for(int i=0; i<N; i++) {
        std::cout << " " << b[i] << " ";
    }
    std::cout << "]" << std::endl;

    // copy input to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // add<<<2,8>>>(dev_a, dev_b, dev_c);
    dot_prod<<<N/THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "A dot B = C = " << *c << std::endl;

    free(a); free(b); free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaDeviceReset();

    return 0;
}
