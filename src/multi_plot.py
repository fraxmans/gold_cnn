import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

import math
import sys
from multiprocessing import Pool
from functools import partial

seq_len = int(sys.argv[1])
kind = "raw"
prefix = "%s/%dmin" % (kind, seq_len)

def load_data():
    data = np.loadtxt("data/GOLD1.csv", delimiter=",", dtype=np.str)
    data = data[:, 2:-1].astype(np.float64)

    return data

def plot(data, interval):

    for i in range(interval[0], interval[1]):
        plt.figure(figsize=(0.3, 0.3))
        plt.axis("off")

        start = i - seq_len + 1
        end = i + 1

        plt.plot(data[start:end, -1])
        fileName = "gen/%s/%d.png" % (prefix, i)
        plt.savefig(fileName)
        plt.close()

    print("Done gen/%s/%d~%d.png" % (prefix, interval[0], interval[1]))

def main():
    data = load_data()
    
    #slicing interval
    batch_size = 256
    sliced_idx = []
    for i in range(seq_len-1, data.shape[0], batch_size):
        sliced_idx.append(i)
    sliced_idx.append(data.shape[0])

    #making subprocess arg
    subprocess_arg = []
    for i in range(len(sliced_idx)-1):
        subprocess_arg.append([sliced_idx[i], sliced_idx[i+1]])

    p = Pool(12)
    func = partial(plot, data)
    p.map(func, subprocess_arg)
    p.close()
    p.join()

main()
