
import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))

def binary_crossentropy(y, y_hat):
    """
    Calculates Binary Cross Entropy cost given ground truth label and prob(1)
    ---------
    Arguments
    ---------
    y [np.array]: Numpy array of shape (m,) denoting ground truth label (0 or 1)
    y_hat [np.array]: Numpy array of shape (m,) denoting probability of 1 for each entry
    -------
    Returns
    -------
    avg_loss [float]: Average BCE cost
    """
    y_hat[y_hat<(1e-6)] = 1e-6
    y_hat[y_hat>(1-1e-6)] = 1-1e-6
    losses = - (y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
    avg_loss = losses.mean()
    return avg_loss

def threshold(y_hat, thresh=0.5):
    """
    Thresholds a probability vector to binarize output
    ---------
    Arguments
    ---------
    y [np.array]: Numpy array of shape (m,) denoting ground truth label (0 or 1)
    y_hat [np.array]: Numpy array of shape (m,) denoting probability of 1 for each entry
    -------
    Returns
    -------
    avg_loss [float]: Average BCE cost
    """
    y_hat_thresh = np.zeros_like(y_hat)
    y_hat_thresh[y_hat>thresh] = 1
    return y_hat_thresh


class LogisticRegression:
    """Logistic Regression implementation"""
    def __init__(self, lr=1e-2, num_iters=1000, gradient_descent_mode="sgd", logging=False):
        """
        Constructor of LogisticRegression class
        ---------
        Arguments
        ---------
        lr [float]: Learning rate
        num_iters [int]: Max number of iterations to run gradient descent
        gradient_descent_mode [str]: Mode of gradient descent. Can be either "batch_gd" 
                                     for batch GD or "sgd" for SGD
        logging [bool]: Whether to print costs between iterations
        -------
        Returns
        -------
        None
        """
        self.lr = lr
        self.num_iters = num_iters
        self.gradient_descent_mode = gradient_descent_mode
        self.logging = logging

    def train(self, X_train, y_train):
        """
        Trains logistic regression model
        ---------
        Arguments
        ---------
        X_train [np.array]: Numpy array of shape (m,d) denoting train dataset X
        y_train [np.array]: Numpy array of shape (m,) denoting train dataset Y containing (0,1)
        -------
        Returns
        -------
        cost_history [list<float>]: List containing iteration-wise costs. Useful for debugging
        """
        X_train = np.concatenate([X_train, np.ones((len(X_train),1))], axis=1)
        n_samples, n_dims = X_train.shape
        self.weight = np.random.randn(n_dims,)
        self.weight[-1] = 0
        cost_history = []
        for iter in range(self.num_iters):
            y_hat = sigmoid(X_train @ self.weight)
            cost = binary_crossentropy(y_train, y_hat)
            if self.gradient_descent_mode=="batch_gd":
                gradient = - 1/n_samples * X_train.T @ (y_train - y_hat)
                self.weight = self.weight - self.lr*gradient
            elif self.gradient_descent_mode=="sgd":
                for i in range(n_samples):
                    y_train_sample, y_hat_sample, X_train_sample = y_train[i],\
                                                          y_hat[i],X_train[i]
                    gradient = - (y_train_sample - y_hat_sample) * X_train_sample
                    self.weight = self.weight - self.lr*gradient
            else:
                raise ValueError("Gradient Descent mode invalid")
            if self.logging:
                print("Iteration: {} Cost: {}".format(iter, cost))
            if self.gradient_descent_mode=="batch_gd" and iter>1 and np.isclose(cost,\
                                                  cost_history[-1], atol=1e-5):
                break
            cost_history.append(cost)
        return cost_history

    def predict_sample(self, sample, thresholding=False):
        """
        Runs prediction for a single sample
        ---------
        Arguments
        ---------
        sample [np.array]: Numpy array of shape (m,) denoting single sample
        thresholding [bool]: Whether to threshold output to (0,1)
        -------
        Returns
        -------
        y_hat [float]: Prob(1) for sample. Output is thresholded if thresholding=True 
        """
        sample = np.concatenate([sample, [1]])
        y_hat = sigmoid(sample @ self.weight)
        if thresholding:
            y_hat = threshold(y_hat)
        return y_hat

    def predict(self, X_test, thresholding=False):
        """
        Runs prediction for a single sample
        ---------
        Arguments
        ---------
        X_test [np.array]: Numpy array of shape (m',d) denoting test dataset X
        thresholding [bool]: Whether to threshold output to (0,1)
        -------
        Returns
        -------
        y_hat [np.array]: Numpy array of shape (m',) denoting element-wise prob(1).
                          Output is thresholded if thresholding=True 
        """
        X_test = np.concatenate([X_test, np.ones((len(X_test),1))], axis=1)
        y_hat = sigmoid(X_test @ self.weight)
        if thresholding:
            y_hat = threshold(y_hat)
        return y_hat
