
import matplotlib.pyplot as plt 
import numpy as np 
import csv 

def plot_values(means, means1, means2, dims):
    # yerr plot the stds, 0 means no stds
    # x = dims
    x = np.log(dims)
    # print(x)
    plt.errorbar(x, means, yerr=0, fmt='-o', label='Privatization')
    plt.errorbar(x, means1, yerr=0, fmt='-o',label='Lock')
    plt.errorbar(x, means2, yerr=0, fmt='-o', label='Query')
    plt.xticks(x,dims)
    plt.legend(loc='best')
    plt.title('Test on queries number')
    plt.ylabel('Speed Up')
    plt.xlabel('Testing values')
    plt.show()

if __name__ == "__main__":
    means_d, means_dl, means_q, means_s = [], [], [], []
    stds_d, stds_dl, stds_q, stds_s = [], [], [], []
    dims = []
    with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-NUM_QUERIES/CPP_parallel_lv_data_NUM_QUERIES.csv') as csvFile:
    #with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-LEN_SEQ/CPP_parallel_lv_data_LEN_SEQ.csv') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter = ',')
        for line in csvFile:
            m, s, d = line.replace('\n','').split(',')
            means_d.append(float(m)/1e+06)
            stds_d.append(float(s)/1e+06)
            dims.append(float(d))

    with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-NUM_QUERIES/CPP_parallel_lv_data_with_lock_NUM_QUERIES.csv') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter = ',')
        for line in csvFile:
            m, s, d = line.replace('\n','').split(',')
            means_dl.append(float(m)/1e+06)
            stds_dl.append(float(s)/1e+06)

    with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-NUM_QUERIES/CPP_parallel_lv_query_NUM_QUERIES.csv') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter = ',')
        for line in csvFile:
            m, s, d = line.replace('\n','').split(',')
            means_q.append(float(m)/1e+06)
            stds_q.append(float(s)/1e+06)

    with open('/home/nicco/Documents/Progetti/ParallelComp-Projects/Pattern_Recognition/Result/test-NUM_QUERIES/CPP_sequential_NUM_QUERIES.csv') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter = ',')
        for line in csvFile:
            m, s, d = line.replace('\n','').split(',')
            means_s.append(float(m)/1e+06)
            stds_s.append(float(s)/1e+06)

    
    means_d = np.asarray(means_s) / np.asarray(means_d)
    means_dl = np.asarray(means_s) / np.asarray(means_dl)
    means_q = np.asarray(means_s) / np.asarray(means_q)

    # print(means_d)
    # print(dims)
    # print(len(stds_d))
    plot_values(means_d, means_dl, means_q, dims)



