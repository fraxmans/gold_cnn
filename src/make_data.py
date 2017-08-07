import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import h5py

import sys

seq_len = int(sys.argv[1])
label_num = int(sys.argv[2])
kind = "raw"
prefix = "%s/%dmin-%dlabel" % (kind, seq_len, label_num)

def load_raw_data():
    raw_data = np.loadtxt("data/GOLD1.csv", delimiter=",", dtype=np.str)
    raw_data = raw_data[:, 2:-1].astype(np.float64)

    return raw_data

def compute_diffs(data):
    diffs = []

    for i in range(1, data.shape[0]):
        diff = data[i][-1] - data[i-1][-1]
        percent = diff/data[i-1][-1]
        diffs.append(percent)

    diffs = np.array(diffs)
 
    return diffs

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
 
def load_pic(raw_data_num):
    pics = []
    for i in range(seq_len-1, raw_data_num):
        path = "gen/%s/%dmin/%d.png" % (kind, seq_len, i)
        img = mpimg.imread(path)
        img = np.mean(img[:, :, 0:4], axis=2)
        pics.append(img)

        if(i % 10000 == 0):
            print("Done %s" % path)

    pics = np.array(pics)

    return pics
    
def write_HDF5(pics, label, name):
    path = "data/%s/%s.hdf5" % (prefix, name)
    dataset = h5py.File(path)
    dataset.create_dataset("data", data=pics)
    dataset.create_dataset("label", data=label)
    dataset.close()

def unison_shuffled_copies(raw_data, diffs, pics):
    assert (raw_data.shape[0] == diffs.shape[0]) and (raw_data.shape[0] == pics.shape[0]) 

    p = np.random.permutation(pics.shape[0])
    
    return raw_data[p], diffs[p], pics[p]

def compute_bounds(train_diffs):
    tmp = np.copy(train_diffs)
    tmp = np.sort(tmp)
    bounds = np.linspace(0, tmp.shape[0], num=label_num+1, dtype=np.int)#n + 1 point -> n range
    bounds[-1] -= 1
    bounds = tmp[bounds]

    return bounds
    
def main():
    raw_data = load_raw_data()
    raw_data_num = raw_data.shape[0]
    pics = load_pic(raw_data_num)
    diffs = compute_diffs(raw_data)

    #remove not-ploted pair 
    raw_data = raw_data[seq_len-1:-1]
    diffs = diffs[seq_len-1:]
    pics = pics[:-1]

    #shuffle
    raw_data, diffs, pics = unison_shuffled_copies(raw_data, diffs, pics)
    
    #split train and test set
    offset = int(diffs.shape[0] * 0.9)
    train_raw_data, test_raw_data = raw_data[0:offset], raw_data[offset:]
    train_diffs, test_diffs = diffs[0:offset], diffs[offset:]
    train_pics, test_pics = pics[0:offset], pics[offset:]

    #compute train_diffs bounds
    bounds = compute_bounds(train_diffs)

    #apply bounds to train and test diff
    train_label = make_label(train_diffs, bounds)
    test_label = make_label(test_diffs, bounds)

    #write to file    
    write_HDF5(train_pics, train_label, "trainset")
    write_HDF5(test_pics, test_label, "testset")
    
main()
