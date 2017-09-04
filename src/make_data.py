import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py

from datetime import datetime
import sys
import os

seq_len = int(sys.argv[1])
label_num = int(sys.argv[2])
kind = "raw"
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
    tmp = np.copy(train_diffs[:, idx_close])
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
            if(diff[idx_close] >= bounds[i] and diff[idx_close] < bounds[i+1]):
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
    gold_raw_data = load_raw_data()

    diffs = compute_diffs(gold_raw_data)

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
main()   
    
