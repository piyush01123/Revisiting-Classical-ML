
import numpy as np

def mse_error(Y_hat, Y):
    """Calculates Mean Squared Error between ground truth Y and predicted Y_hat"""
    n = len(Y)
    diff = Y-Y_hat
    return 1/n*(diff@diff)

def mae_error(Y_hat, Y):
    """Calculates Mean Average Error between ground truth Y and predicted Y_hat"""
    n = len(Y)
    diff = Y-Y_hat
    return 1/n*np.sum(np.abs(diff))

def r_squared(Y_hat, Y):
    """Calculates R squared between ground truth Y and predicted Y_hat"""
    n = len(Y)
    Y_mean = np.sum(Y)/n
    diff1 = Y-Y_hat
    diff2 = Y-Y_mean
    rss = diff1@diff1
    tss = diff2@diff2
    return 1-(rss/tss)