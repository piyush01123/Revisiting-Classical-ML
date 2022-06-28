
import numpy as np


def flatten_and_normalize_data(X_train, X_test, ndims, mode="standard"):
    """
    Flattens and normalize a dataset
    ---------
    Arguments
    ---------
    model_choice [str]: Whether to run KMeans or KMeans++
    X_train [np.array]: Numpy array of shape (m,d) denoting train dataset
    X_test [np.array]: Numpy array of shape (m,d) denoting test dataset
    ndims [int]: Number of dimensions of dataset
    mode [str]: Normalization mode. One of "standard" or "min_max"
    -------
    Returns
    -------
    X_train [np.array]: Numpy array of shape (m,d) denoting normalized train dataset
    X_test [np.array]: Numpy array of shape (m,d) denoting normalized test dataset
    """
    X_train = X_train.reshape((-1,ndims))
    if mode=="min_max":
        mins, maxes = X_train.min(0), X_train.max(0)
        X_train = (X_train-mins)/((maxes-mins)+1e-6)
        X_test = X_test.reshape((-1,ndims))
        X_test = (X_test-mins)/((maxes-mins)+1e-6)
    elif mode=="standard":
        means, stds = X_train.mean(0), X_train.std(0)
        X_train = (X_train-means)/(stds+1e-6)
        X_test = X_test.reshape((-1,ndims))
        X_test = (X_test-means)/(stds+1e-6)
    else:
        raise ValueError("Invalid mode.")
    return X_train, X_test
