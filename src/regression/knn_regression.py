import numpy as np
from scipy.spatial import distance


class KNN_Regressor:
    """KNN Regressor class"""
    def __init__(self, Xt, Yt, normalize=True):
        """
        Constructor of KNN regressor
        Arguments
        ---------
        Xt (np.array): Training X array
        Yt (np.array): Training Y array
        normalize (bool): Whether to normalize or not        
        """
        self.x_train = Xt
        self.y_train = Yt
        self.normalize=normalize
        if self.normalize:
            self.stats = self.save_stats()
            self.x_train = self.scale(self.x_train)

    def save_stats(self):
        """Save statistics of training dataset"""
        return {"maxes": self.x_train.max(axis=0), \
                "mins": self.x_train.min(axis=0), \
                "means": self.x_train.mean(axis=0), \
                "stds": self.x_train.std(axis=0), \
               }

    def scale(self, dataset, mode="min_max"):
        """Scale dataset as per strategy given by 'mode'"""
        if mode=="min_max":
            return (dataset-self.stats["mins"])/(self.stats["maxes"]-self.stats["mins"])
        elif mode=="gaussian":
            return (dataset-self.stats["means"])/self.stats["stds"]

    def predict(self, X, K, metric='euclidean', weighting=False):
        """
        Calculate distance matrix, nearest neighbors and return predictions given test set X 
        and number of neighbors K

        Arguments
        ---------
        X (np.array): Test Set
        K (int): Number of neighbors to use for KNN
        metric (str): Distance measure to use
        Returns
        -------
        np.array: Predicted Y
        """
        if self.normalize:
            X = self.scale(X)
        distance_matrix = distance.cdist(X, self.x_train, metric)
        nearest_neighbors = distance_matrix.argsort(axis=1)[:,:K]
        if not weighting:
            return self.y_train[nearest_neighbors].mean(axis=1)
        else:
            weights = 1/np.arange(1,K+1)
            weights = weights/weights.sum()
            return self.y_train[nearest_neighbors]@weights   

