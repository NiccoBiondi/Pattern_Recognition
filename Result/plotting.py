
import matplotlib.pyplot as plt 
import numpy as np 
import csv 

def plot_values(means, stds):
    plt.plot (dims, means, '-o')
    plt.fill_between(dims,means-stds,means+stds,alpha=.1)
    plt.show()

if __name__ == "__main__":
    means = []
    stds = []
    dims = []
    with open('naive.csv') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter = ',')
        for line in csvFile:
            m, s, d = line.replace('\n','').split(',')
            means.append(float(m))
            stds.append(float(s))
            dims.append(int(d))
    # print(means)
    # print(stds)

    plot_values(np.asarray(means), np.asarray(stds))



