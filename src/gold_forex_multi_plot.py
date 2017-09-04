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
kind = "gold_forex"
prefix = "%s/%dmin" % (kind, seq_len)

idx_time = 0
idx_close = 1

def load_raw_data():
    gold_raw_data = np.loadtxt("data/XAUUSD_small.csv", delimiter=" ")
    gold_raw_data = gold_raw_data[:, [0, -1]]
    gold_raw_data = remove_duplicate(gold_raw_data)

    forex_raw_data = np.loadtxt("data/EURUSD_small.csv", delimiter=" ")
    forex_raw_data = forex_raw_data[:, [0, -1]]
    forex_raw_data = remove_duplicate(forex_raw_data)

    return gold_raw_data, forex_raw_data

def remove_duplicate(data):
    
    _, idx = np.unique(data[:, 0], return_index=True)
    data = data[idx]
   
    return data

def compute_diffs(data):
    diffs = []

    for i in range(seq_len, data.shape[0]):
        start = i - seq_len
        end = i

        if(data[end, idx_time] - data[start, idx_time] != seq_len * 60.0):
            continue

        diff = data[i][idx_close] - data[i-1][idx_close]
        percent = diff/data[i-1][idx_time]
        diffs.append(percent)

    diffs = np.array(diffs)

    return diffs
                                               
def relative_gold_forex(gold_raw_data, forex_raw_data):

    not_match_cnt = 0
    idxs = []

    for i in range(gold_raw_data.shape[0]):
        idx = search(gold_raw_data[i, idx_time], forex_raw_data) 
        if(idx != None):
            gold_raw_data[i, idx_close] /= forex_raw_data[idx, idx_close]
            idxs.append(i)
        else:
            not_match_cnt += 1

    return not_match_cnt, gold_raw_data[idxs], np.array(idxs)

def search(target_time, forex_raw_data):
    start = 0
    end = forex_raw_data.shape[0] - 1

    while(start <= end):
        mid = (start + end) // 2
        current_time = forex_raw_data[mid, idx_time]

        if(current_time == target_time):
            return mid
        if(current_time < target_time):
            start = mid + 1
        else:
            end = mid - 1

def plot(data, interval):

    for i in range(interval[0], interval[1]):
        start = i - seq_len 
        end = i 

        if(data[end, idx_time] - data[start, idx_time] != seq_len * 60.0):
            continue

        plt.figure(figsize=(0.3, 0.3))
        plt.axis("off")
        plt.plot(data[start:end, -1])
        fileName = "gen/%s/%d.png" % (prefix, i)
        plt.savefig(fileName)
        plt.close()

    print("[%s] Done gen/%s/%d~%d.png" % (datetime.now(), prefix, interval[0], interval[1]))

def load_pic(raw_data_num):
    pics = []
    for i in range(seq_len, raw_data_num):
        path = "gen/%s/%d.png" % (prefix, i)
        if(not os.path.isfile(path)):
            continue
        
        img = mpimg.imread(path)
        img = np.mean(img[:, :, 0:4], axis=2)
        pics.append(img)

        if(i % 10000 == 0):
            print("Done %s" % path)

    pics = np.array(pics)
    return pics

def compute_bounds(train_diffs):
    tmp = np.copy(train_diffs)
    tmp = np.sort(tmp)
    bounds = np.linspace(0, tmp.shape[0], num=label_num+1, dtype=np.int)#n + 1 point -> n range
    bounds[-1] -= 1
    bounds = tmp[bounds]

    return bounds

def make_label(diffs, bounds):
    label = []

    for diff in diffs:
        find = False
        for i in range(bounds.shape[0]-1):
            if(diff >= bounds[i] and diff < bounds[i+1]):
                label.append(i)
                find = True
                break
        #this solve the max diff won't append label
        if(find == False):
            label.append(seq_len-1)

    label = np.array(label)
    return label

def unison_shuffled_copies(pics, diffs):
    print(pics.shape[0], diffs.shape[0])
    assert (pics.shape[0] == diffs.shape[0]) 

    p = np.random.permutation(pics.shape[0])

    return pics[p], diffs[p]

def write_HDF5(pics, label, name):
    path = "data/%s-%dlabel/%s.hdf5" % (prefix, label_num, name)
    dataset = h5py.File(path)
    dataset.create_dataset("data", data=pics)
    dataset.create_dataset("label", data=label)
    dataset.close()

def main():
    gold_raw_data, forex_raw_data = load_raw_data()
    gold_raw_bkp = np.copy(gold_raw_data)

    #compute relative close price with time-matched data
    not_match_cnt, gold_raw_data, mask = relative_gold_forex(gold_raw_data, forex_raw_data)

    gold_raw_bkp = gold_raw_bkp[mask]
    diffs = compute_diffs(gold_raw_bkp)
    print(gold_raw_data.shape[0], diffs.shape[0])

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
    func = partial(plot, gold_raw_data)
    p.map(func, subprocess_arg)
    p.close()
    p.join()

    print("There are %d data not match between gold and forex" % not_match_cnt)

    """
    #make train and test dataset
    pics = load_pic(gold_raw_data.shape[0])
    pics, diffs = unison_shuffled_copies(pics, diffs)
    
    offset = int(0.9 * diffs.shape[0])
    train_diffs, test_diffs = diffs[:offset], diffs[offset:]
    bounds = compute_bounds(train_diffs) 

    train_data, test_data = pics[:offset], pics[offset:]
    train_label = make_label(train_diffs, bounds)
    test_label = make_label(test_diffs, bounds)
    
    #write to hdf5 file
    write_HDF5(train_data, train_label, "trainset")
    write_HDF5(test_data, test_label, "testset")
    """
main()   
    
