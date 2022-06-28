

import os
import gzip
import numpy as np


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

