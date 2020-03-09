
import matplotlib.pyplot as plt 
import numpy as np 
import csv 

def plot_values(means, stds):
    plt.plot (means)
    plt.fill_between(range(len(means)),means-stds,means+stds,alpha=.1)
    plt.show()

if __name__ == "__main__":
    means = []
    stds = []
    with open('prova.csv') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter = ',')
        for line in csvFile:
            m, s = line.replace('\n','').split(',')
            means.append(float(m))
            stds.append(float(s))
    # print(means)
    # print(stds)

    plot_values(np.asarray(means), np.asarray(stds))



