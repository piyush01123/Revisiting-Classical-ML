
import numpy as np


class LinearRegression:
    """Linear Regression class [Closed Form Solution]"""
    def __init__(self, x_train, y_train, normalize=True):
        """
        Constructor of Linear regression class
        
        Arguments
        ---------
        x_train (np.array): Training data X
        y_train (np.array): Training data Y
        normalize (bool): Whether to normalize the data
        """
        self.normalize=normalize
        if self.normalize:
            self.stats = {"maxes": x_train.max(axis=0), \
                          "mins": x_train.min(axis=0), \
                          "means": x_train.mean(axis=0), \
                          "stds": x_train.std(axis=0), \
                         }
            x_train = self.scale(x_train)
        x_train = np.hstack([x_train, np.ones((len(x_train),1))])
        self.weights = np.linalg.inv(x_train.T@x_train)@x_train.T@y_train

    def scale(self, dataset, mode="gaussian"):
        """Scales the dataset as per given mode"""
        if mode=="min_max":
            return (dataset-self.stats["mins"])/(self.stats["maxes"]-self.stats["mins"])
        elif mode=="gaussian":
            return (dataset-self.stats["means"])/self.stats["stds"]

    def predict(self, X):
        """Returns prediction for test dataset X"""
        if self.normalize:
            X = self.scale(X)
        X = np.hstack([X, np.ones((len(X),1))])
        return X@self.weights
