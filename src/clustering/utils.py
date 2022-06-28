

import os
import gzip
import numpy as np
import pandas as pd


def load_mnist(path, kind):
    """Load MNIST data from `path`"""
    kind="t10k" if kind=="val" else "train"
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
    

def read_latent_representation(data_csv_path="data.csv"):
    """
    Handles I/O for latent representation. Assumes that "data.csv" is present
    at working directory. If not present then run the cell above.
    ---------
    Arguments
    ---------
    data_csv_path [str]: Path of "data.csv" file
    -------
    Returns
    -------
    data_X [np.array]: Numpy array of shape (m,d) denoting dataset X
    data_Y [np.array]: Numpy array of shape (m,) denoting dataset Y
    """
    fp_x = data_csv_path.replace("data.csv","data_X.csv")
    fp_y = data_csv_path.replace("data.csv","data_Y.csv")
    if os.path.isfile(fp_x) and os.path.isfile(fp_y):
        pass
    else:
        f = open(data_csv_path,'r')
        lines = f.read().split('\n')
        lines_X = [line[1:line.rindex(']')] for line in lines[:-1]]
        lines_Y = [line[line.rindex(']')+3:] for line in lines[:-1]]
        f = open(fp_x, 'w')
        f.write('\n'.join(lines_X))
        f = open(fp_y,'w')
        f.write('\n'.join(lines_Y))

    data_X = np.genfromtxt(fp_x, delimiter=',')
    data_Y = pd.read_csv(fp_y, header=None).to_numpy().reshape(-1,)
    label_dict = {j: i for i, j in enumerate(np.unique(data_Y))}
    data_Y = np.array([label_dict[i] for i in data_Y])
    return data_X, data_Y


def flatten_and_normalize_data(X_train, X_test, ndims):
    """
    Flattens and normalize a dataset
    ---------
    Arguments
    ---------
    X_train [np.array]: Numpy array of shape (m,d) denoting train dataset
    X_test [np.array]: Numpy array of shape (m,d) denoting test dataset
    ndims[int]: Number of dimensions of dataset
    -------
    Returns
    -------
    X_train [np.array]: Numpy array of shape (m,d) denoting normalized train dataset
    X_test [np.array]: Numpy array of shape (m,d) denoting normalized test dataset
    """
    X_train = X_train.reshape((-1,ndims))
    mins, maxes = X_train.min(0), X_train.max(0)
    X_train = (X_train-mins)/((maxes-mins)+1e-6)

    X_test = X_test.reshape((-1,ndims))
    X_test = (X_test-mins)/((maxes-mins)+1e-6)

    return X_train, X_test

