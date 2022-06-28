
import numpy as np


def my_accuracy(y_gt, y_pred):
    """
    My implementation of accuracy calculation
    ---------
    Arguments
    ---------
    y_gt[np.array]: Numpy array of shape (m,)
    y_pred[np.array]: Numpy array of shape (m,)
    -------
    Returns
    -------
    [float]: Accuracy
    """
    return (y_gt==y_pred).sum()/len(y_gt)

def my_confusion_matrix(y_gt, y_pred, n_classes):
    """
    My implementation of confusion matrix calculation
    ---------
    Arguments
    ---------
    y_gt[np.array]: Numpy array of shape (m,)
    y_pred[np.array]: Numpy array of shape (m,)
    n_classes[int]: Number of classes
    -------
    Returns
    -------
    conf_mat[np.array]: Numpy array of shape (n_classes,n_classes) denoting confusion matrix
    """
    conf_mat = np.zeros((n_classes,n_classes),dtype=int)
    for i in range(n_classes):
    for j in range(n_classes):
        conf_mat[i,j] = np.bitwise_and(y_gt==i, y_pred==j).sum()
    return conf_mat
