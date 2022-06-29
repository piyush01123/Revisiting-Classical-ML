
import numpy as np
import scipy.stats


class NaiveBayes:
    def __init__(self, n_classes):
        self.n_classes = n_classes
    def train(self, X_train, y_train):
        self.y_freq = {y: sum(y_train==y) for y in range(self.n_classes)}
        self.X_stats = {y: {"X_mean": np.mean(X_train[y_train==y], axis=0), \
                    "X_std": np.std(X_train[y_train==y], axis=0) \
                    } \
                for y in range(self.n_classes)
                }
    def predict(self, X_val):
        P = np.zeros((X_val.shape[0], self.n_classes))
        for row, x in enumerate(X_val):
            probs = []
            for y in range(self.n_classes):
                mean, std = self.X_stats[y]["X_mean"], self.X_stats[y]["X_std"]
                A = scipy.stats.norm(mean, std).pdf(x)
                B = np.where(np.all([x==mean, std==0], axis=0), 1, A)
                C = np.where(np.all([x!=mean, std==0], axis=0), 1e-4, B)
                probs.append(np.product(C)*self.y_freq[y])
            P[row] = probs
        pred = np.argmax(P, axis=1)
        return pred
