
import numpy as np
from scipy.spatial import distance


class KNN_Classifier(object):
    """ a kNN classifier with L2 distance """
    def __init__(self):
        """Constructor (empty)"""
        pass
    
    def train(self, x_train, y_train, normalize=False,norm_mode="pixel"):
        """"
        "trains" the KNN; in reality there is no training going on, it just remember the training data
        Arguments
        ---------
        x_train (np.array): Training data X
        y_train (np.array): Training data Y
        normalize (bool): Whether to normalize the data or not
        norm_mode (str): Whether to use pixel position normalization or global normalization
        """
        self.x_train = x_train
        self.y_train = y_train
        self.normalize = normalize
        self.norm_mode = norm_mode
        if self.normalize:
            self.stats = self.save_stats()
            self.x_train = self.scale(self.x_train, self.norm_mode)                

    def save_stats(self):
        """Saves the pixel wise and global statistics of data"""
        return {"maxes": self.x_train.max(axis=0), \
                "mins": self.x_train.min(axis=0), \
                "means": self.x_train.mean(axis=0), \
                "stds": self.x_train.std(axis=0), \
                "min_overall": self.x_train.min(),\
                "max_overall": self.x_train.max(),\
                "mean_overall": self.x_train.mean(),\
                "std_overall": self.x_train.std()
               }

    def scale(self, dataset, norm_mode, mode="min_max"):
        """Given a a dataset and a normalization mode, scales it"""
        if norm_mode=="pixel":
            if mode=="min_max":
                return (dataset-self.stats["mins"])/(self.stats["maxes"]-self.stats["mins"])
            elif mode=="gaussian":
                return (dataset-self.stats["means"])/self.stats["stds"]
        else:
            if mode=="min_max":
                return (dataset-self.stats["min_overall"])/(self.stats["max_overall"]-self.stats["min_overall"])
            elif mode=="gaussian":
                return (dataset-self.stats["mean_overall"])/self.stats["std_overall"]


    def compute_distances_two_loops(self,X):
        """Computes distances between X and training X in 2 loops"""
        D = np.zeros((len(X),len(self.x_train)))
        for i in range(len(X)):
            for j in range(len(self.x_train)):
                diff = X[i]-self.x_train[j]
                D[i,j] = np.sqrt(diff@diff)
        return D
    
    def compute_distances_one_loop(self,X):
        """Computes distances between X and training X in 1 loop"""
        D = np.zeros((len(X),len(self.x_train)))
        for i in range(len(X)):
            diff = X[i]-self.x_train
            D[i] = np.sqrt((diff**2).sum(1))
        return D

    def compute_distances_no_loop(self,X):
        """Computes distances between X and training X without any loop"""
        if self.normalize:
            X = self.scale(X, self.norm_mode)
        # return np.sqrt(((X[:, :, None] - self.x_train[:, :, None].T) ** 2).sum(1))
        return distance.cdist(X, self.x_train, "euclidean")

    def predict_labels(self,D,K):
        nearest_neighbors = D.argsort(axis=1)[:,:K]
        label_count_nbrs = np.array([[sum(row==i) for i in range(10)] for row in self.y_train[nearest_neighbors]])
        return label_count_nbrs.argmax(1)
