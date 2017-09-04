import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py

from datetime import datetime
import sys
import os
from multiprocessing import Pool
from functools import partial

seq_len = int(sys.argv[1])
label_num = int(sys.argv[2])
kind = "raw-avg"
prefix = "%s/%dmin" % (kind, seq_len)

idx_time = 0
idx_close = 1

def load_raw_data():
    gold_raw_data = np.loadtxt("data/XAUUSD.csv", delimiter=" ")
    gold_raw_data = gold_raw_data[:, [0, -1]]
    gold_raw_data = remove_duplicate(gold_raw_data)

    return gold_raw_data

def remove_duplicate(data):
    
    _, idx = np.unique(data[:, 0], return_index=True)
    data = data[idx]
   
    return data

def compute_diffs(data):
    interval_avg = compute_interval_avg(data)

    diffs = []
    for i in range(0, interval_avg.shape[0]-seq_len):
        start = i
        end = i + seq_len

        if(interval_avg[end, idx_time] - interval_avg[start, idx_time] != seq_len * 60.0):
            continue

        diff = interval_avg[end, idx_close] - interval_avg[start, idx_close]
        percent = diff / interval_avg[start, idx_close]
        diffs.append([interval_avg[start, idx_time], percent])

    diffs = np.array(diffs)
    return diffs

def compute_interval_avg(data):
    avgs = []

    for i in range(seq_len, data.shape[0]):
        start = i - seq_len
        end = i

        if(data[end, idx_time] - data[start, idx_time] != seq_len * 60):
            continue

        avg = np.mean(data[start:end, idx_close])
        avgs.append([data[end, idx_time], avg])

    avgs = np.array(avgs)
    return avgs

def plot(gold_raw_data, diffs, interval):
        
    plt.figure(figsize=(0.3, 0.3))
    for i in range(interval[0], interval[1]):
        plt.axis("off")

        start = i - seq_len
        end = i 

        #plot with matched timestamp
        isFind = search(gold_raw_data[end, idx_time], diffs)
        if(isFind == None):
            continue

        plt.plot(gold_raw_data[start:end, idx_close])
        fileName = "gen/%s/%d.png" % (prefix, i)
        plt.savefig(fileName)
        plt.clf()

    print("Done %s/%d.jpg-%d.jpg" % (prefix, interval[0], interval[1]))

def search(target_time, diffs):
    start = 0
    end = diffs.shape[0] - 1

    while(start <= end):
        mid = (start + end) // 2
        current_time = diffs[mid, idx_time]

        if(current_time == target_time):
            return mid
        if(current_time < target_time):
            start = mid + 1
        else:
            end = mid - 1
                                                    
def main():
    gold_raw_data = load_raw_data()

    diffs = compute_diffs(gold_raw_data)
    print("gold len: %d\tdiffs len: %d" % (gold_raw_data.shape[0], diffs.shape[0]))

    #slicing interval
    batch_size = 16384 
    sliced_idx = []
    for i in range(seq_len, gold_raw_data.shape[0], batch_size):
        sliced_idx.append(i)
    sliced_idx.append(gold_raw_data.shape[0])

    #making subprocess arg
    subprocess_arg = []
    for i in range(len(sliced_idx)-1):
        subprocess_arg.append([sliced_idx[i], sliced_idx[i+1]])

    p = Pool(12)
    func = partial(plot, gold_raw_data, diffs)
    p.map(func, subprocess_arg)
    p.close()
    p.join()

main()   
    
