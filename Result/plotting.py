
import matplotlib.pyplot as plt 
import numpy as np 
import csv 

def plot_values(means, means1, means2, means3, dims):
    # print(len(means[0]))
    # print(type(dims[0]))
    # yerr plot the stds, 0 means no stds
    # x = int(dims)
    x = np.log(dims)
    print(means[len(dims):])
    plt.errorbar(x, means[:len(dims)], yerr=0, fmt='-o', label='Constant')
    plt.errorbar(x, means1[:len(dims)], yerr=0, fmt='-o',label='Naive')
    plt.errorbar(x, means2[:len(dims)], yerr=0, fmt='-o', label='Private')
    plt.errorbar(x, means3[:len(dims)], yerr=0, fmt='-o', label='Tiling')
    plt.xticks(x,dims)
    plt.legend(loc='upper left')
    #plt.title('Test on threads number')
    plt.ylabel('Computational time [μs]')
    plt.xlabel('Testing values [· e+05]')
    # plt.savefig('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/images/num_threads.png')
    plt.show()

if __name__ == "__main__":
    means_d, means_dl, means_q, means_s = [], [], [], []
    stds_d, stds_dl, stds_q, stds_s = [], [], [], []
    dims = []
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_PATTERN_SEQ/CPP_parallel_lv_data_LEN_PATTERN_SEQ.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-n_thread/CPP_parallel_lv_data_nthreads.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-NUM_QUERIES/CPP_parallel_lv_data_NUM_QUERIES.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_SEQ/CPP_parallel_lv_data_LEN_SEQ.csv') as csvFile:
    
    with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_SEQ/CUDA_constant_LEN_SEQ.csv') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter = ',')
        for line in csvFile:
            m, s, d = line.replace('\n','').split(',')
            means_d.append(float(m))
            stds_d.append(float(s))
            dims.append((float(d)/1e+05))

    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_PATTERN_SEQ/CPP_parallel_lv_data_with_lock_LEN_PATTERN_SEQ.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-NUM_QUERIES/CPP_parallel_lv_data_NUM_QUERIES.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_SEQ/CPP_parallel_lv_data_lock_LEN_SEQ.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-NUM_QUERIES/CPP_parallel_lv_data_with_lock_NUM_QUERIES.csv') as csvFile:
    
    with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_SEQ/CUDA_naive_LEN_SEQ.csv') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter = ',')
        for line in csvFile:
            m, s, d = line.replace('\n','').split(',')
            means_dl.append(float(m))
            stds_dl.append(float(s))

    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_PATTERN_SEQ/CPP_parallel_lv_query_LEN_PATTERN_SEQ.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-n_thread/CPP_parallel_lv_query_nthreads.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_SEQ/CPP_parallel_lv_query_LEN_SEQ.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-NUM_QUERIES/CPP_parallel_lv_query_NUM_QUERIES.csv') as csvFile:
    
    with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_SEQ/CUDA_private_LEN_SEQ.csv') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter = ',')
        for line in csvFile:
            m, s, d = line.replace('\n','').split(',')
            means_q.append(float(m))
            stds_q.append(float(s))

    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_PATTERN_SEQ/CPP_sequential_LEN_PATTERN_SEQ.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-NUM_QUERIES/CPP_parallel_lv_data_NUM_QUERIES.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_SEQ/CPP_sequential_LEN_SEQ.csv') as csvFile:
    # with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-NUM_QUERIES/CPP_sequential_NUM_QUERIES.csv') as csvFile:

    with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_SEQ/CUDA_tiling_LEN_SEQ.csv') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter = ',')
        for line in csvFile:
            m, s, d = line.replace('\n','').split(',')
            means_s.append(float(m))
            stds_s.append(float(s))

    
    # means_d = np.asarray(means_s) / np.asarray(means_d)
    # means_dl = np.asarray(means_s) / np.asarray(means_dl)
    # means_q = np.asarray(means_s) / np.asarray(means_q)

    # per threads
    # sp_d = [ means_d[0]/means_d[i] for i in range(1,len(means_d))]
    # sp_q = [ means_q[0]/means_q[i] for i in range(1,len(means_q))]
    # dims = dims[1:]

    # print(sp_d, sp_q, dims)

    # sp_d = means_d[1:] / means_d[0]
    # sp_q = means_q[1:] / means_q[0]

    # print(means_d)
    # print(dims)
    # print(len(stds_d))

    # PER CUDA
    c = np.concatenate((means_d, stds_d))
    n = np.concatenate((means_dl, stds_dl))
    p = np.concatenate((means_q, stds_q))
    t = np.concatenate((means_s, stds_s))
    

    plot_values(c,n,p,t, dims)



