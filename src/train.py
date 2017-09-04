import tensorflow as tf
import numpy as np
import h5py

from tflearn import input_data, DNN
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression
from tflearn.activations import softmax, relu
from tflearn.objectives import softmax_categorical_crossentropy

seq_len = 5
label_num = 5
run_id = "raw-avg/%dmin-%dlabel" % (seq_len, label_num)
batch_size = 256
epoch_num = 300
learning_rate = 0.05

def load_data():

    with h5py.File("data/" + run_id + "/trainset.hdf5") as f:
        train_data = f["data"][()].astype(np.float32)
        train_label = f["label"][()]
    train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], train_data.shape[2], 1])
    train_label = (np.arange(label_num) == train_label[:, None]).astype(np.float32)

    with h5py.File("data/" + run_id + "/testset.hdf5") as f:
        test_data = f["data"][()].astype(np.float32)
        test_label = f["label"][()]
    test_data = np.reshape(test_data, [test_data.shape[0], test_data.shape[1], test_data.shape[2], 1])
    test_label = (np.arange(label_num) == test_label[:, None]).astype(np.float32)

    return train_data, train_label, test_data, test_label

def build_net():
    net = input_data(shape=[None, 30, 30, 1])

    net = conv_2d(net, 32, 5, activation="relu")
    net = max_pool_2d(net, 2, 2, padding="valid")
    net = batch_normalization(net)

    net = conv_2d(net, 64, 5, activation="relu")
    net = max_pool_2d(net, 2, 2, padding="valid")
    net = batch_normalization(net)

    net = fully_connected(net, 1024, activation="relu")
    net = batch_normalization(net)

    net = fully_connected(net, label_num)

    net = regression(net, optimizer="adam", loss="softmax_categorical_crossentropy", learning_rate=learning_rate)
    net = DNN(net, tensorboard_dir="log/")

    return net
        

def main():
    train_data, train_label, test_data, test_label = load_data()
    net = build_net()

    net.fit(train_data, train_label, 
            n_epoch=epoch_num, 
            shuffle=True, 
            validation_set=(test_data, test_label),
            show_metric=True,
            batch_size=batch_size,
            run_id=run_id)
    
main()
