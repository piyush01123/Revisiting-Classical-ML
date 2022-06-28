
import numpy as np


class KMeansPlusPlus:
    """
    KMeans++ Clustering
    """
    def __init__(self, distance_fn, num_clusters, num_iters=1000, random_state=1):
        """
        KMeans++ class constructor
        ---------
        Arguments
        ---------
        distance_fn [str]: Distance function used for KMeans++. Can be "euclidean" or "manhattan"
        num_clusters [int]: Number of clusters used for K Means++ Clustering (K value)
        num_iters [int]: Maximum number of iterations to run for KMeans++ Clustering. Often it will converge before this.
        random_state [int]: Random state to use for initialization heuristic
        -------
        Returns
        -------
        None
        """
        if distance_fn not in ["euclidean", "manhattan"]:
            raise ValueError("Invalid distance_fn argument")
        self.distance_fn = distance_fn
        self.num_clusters = num_clusters
        self.num_iters = num_iters
        self.random_state = random_state


    def initialize_centroids_kmeans_plus_plus(self, X_train):
        """
        KMeans++ initialization heuristic
        ---------
        Arguments
        ---------
        X_train [np.array]: Numpy array of shape (m,d) denoting dataset
        -------
        Returns
        -------
        centroids[np.array]: Numpy array of shape (K,d) denoting centroids
        """
        num_samples = X_train.shape[0]
        centroids = np.zeros((self.num_clusters, X_train.shape[1]))
        np.random.RandomState(self.random_state)
        shuffled_indices = np.random.permutation(num_samples)
        centroids[0] = X_train[shuffled_indices[0]]
        for k in range(1,self.num_clusters):
            if self.distance_fn == "euclidean":
                distance_matrix = self.compute_distance_matrix_euclidean(X_train, centroids)
            if self.distance_fn == "manhattan":
                distance_matrix = self.compute_distance_matrix_manhattan(X_train, centroids)
            cluster_assignments = self.compute_cluster_assignments(distance_matrix)
            distance_distribution = np.zeros(X_train.shape[0])
            for k_dash in range(k):
                errors_kth_cluster = np.linalg.norm(X_train[cluster_assignments == k_dash] - \
                                                    centroids[k_dash], axis=1, ord=2)
                distance_distribution[cluster_assignments == k_dash] = np.square(errors_kth_cluster)
            distance_distribution = distance_distribution/distance_distribution.sum()
            choice = np.random.choice(num_samples, 1, p=distance_distribution).item()
            centroids[k] = X_train[choice]
        return centroids


    def compute_centroids(self, X_train, cluster_assignments):
        """
        Computes centroids given dataset and cluster assignments
        ---------
        Arguments
        ---------
        X_train [np.array]: Numpy array of shape (m,d) denoting dataset
        cluster_assignments[np.array]: Numpy array of shape (m,) denoting cluster assignments
        -------
        Returns
        -------
        centroids[np.array]: Numpy array of shape (K,d) denoting centroids
        """
        centroids = np.zeros((self.num_clusters, X_train.shape[1]))
        for k in range(self.num_clusters):
            centroids[k] = X_train[cluster_assignments==k].mean(0)
        return centroids


    def compute_distance_matrix_euclidean(self, X_train, centroids):
        """
        Computes Euclidean distance matrix given dataset and centroids
        ---------
        Arguments
        ---------
        X_train [np.array]: Numpy array of shape (m,d) denoting dataset
        centroids[np.array]: Numpy array of shape (K,d) denoting centroids
        -------
        Returns
        -------
        distance_matrix[np.array]: Numpy array of shape (m,K) denoting Euclidean distance matrix
        """
        distance_matrix = np.zeros((X_train.shape[0], self.num_clusters))
        for k in range(self.num_clusters):
            norms_k_cluster = np.linalg.norm(X_train - centroids[k], axis=1, ord=2)
            distances_k_cluster = np.square(norms_k_cluster)
            distance_matrix[:, k] = distances_k_cluster
        return distance_matrix


    def compute_distance_matrix_manhattan(self, X_train, centroids):
        """
        Computes Manhattan distance matrix given dataset and centroids
        ---------
        Arguments
        ---------
        X_train [np.array]: Numpy array of shape (m,d) denoting dataset
        centroids[np.array]: Numpy array of shape (K,d) denoting centroids
        -------
        Returns
        -------
        distance_matrix[np.array]: Numpy array of shape (m,K) denoting Manhattan distance matrix
        """
        distance_matrix = np.zeros((X_train.shape[0], self.num_clusters))
        for k in range(self.num_clusters):
            distance_matrix[:, k] = np.linalg.norm(X_train - centroids[k], axis=1, ord=1)
        return distance_matrix


    def compute_cluster_assignments(self, distance_matrix):
        """
        Computes cluster assignments given a distance matrix
        ---------
        Arguments
        ---------
        distance_matrix[np.array]: Numpy array of shape (m,K) denoting Manhattan distance matrix
        -------
        Returns
        -------
        cluster_assignments[np.array]: Numpy array of shape (m,) denoting cluster assignments
        """
        cluster_assignments = np.argmin(distance_matrix, axis=1)
        return cluster_assignments


    def compute_SSE(self, X_train, cluster_assignments, centroids):
        """
        Computes sum of squared error given dataset, cluster assignments and centroids
        ---------
        Arguments
        ---------
        X_train [np.array]: Numpy array of shape (m,d) denoting dataset
        centroids[np.array]: Numpy array of shape (K,d) denoting centroids
        cluster_assignments[np.array]: Numpy array of shape (m,) denoting cluster assignments
        -------
        Returns
        -------
        SSE [float]: Scalar denoting SSE
        """
        distances = np.zeros(X_train.shape[0])
        for k in range(self.num_clusters):
            errors_kth_cluster = np.linalg.norm(X_train[cluster_assignments == k] - centroids[k], axis=1, ord=2)
            distances[cluster_assignments == k] = errors_kth_cluster
        SSE = np.sum(np.square(distances))
        return SSE


    def train(self, X_train, logging=False):
        """
        Runs KMeans++ clustering algorithm. Performs initialization and then iterative runs the algorithm.
        ---------
        Arguments
        ---------
        X_train [np.array]: Numpy array of shape (m,d) denoting dataset
        logging [bool]: Whether to print SSE values between iterations
        -------
        Returns
        -------
        None
        """
        if logging:
            print("Running KMeans++ Init")
        self.centroids = self.initialize_centroids_kmeans_plus_plus(X_train)
        self.SSE_iters = []
        for i in range(self.num_iters):
            centroids_curr = self.centroids
            if self.distance_fn == "euclidean":
                distance_matrix = self.compute_distance_matrix_euclidean(X_train, centroids_curr)
            if self.distance_fn == "manhattan":
                distance_matrix = self.compute_distance_matrix_manhattan(X_train, centroids_curr)
            self.cluster_assignments = self.compute_cluster_assignments(distance_matrix)
            self.centroids = self.compute_centroids(X_train, self.cluster_assignments)
            SSE = self.compute_SSE(X_train, self.cluster_assignments, self.centroids)
            if logging:
                print("Iteration:{} SSE:{}".format(i, SSE))
            if i>1 and np.isclose(SSE, self.SSE_iters[-1], atol=1e-6):
                break
            self.SSE_iters.append(SSE)
        self.final_SSE = SSE


    def predict(self, X_test):
        """
        Computers cluster assignments for a new test set after the fitting part is done.
        ---------
        Arguments
        ---------
        X_test [np.array]: Numpy array of shape (m',d) denoting test dataset
        -------
        Returns
        -------
        cluster_assignments[np.array]: Numpy array of shape (m',) denoting cluster assignments
        """
        if self.distance_fn == "euclidean":
            distance_matrix = self.compute_distance_matrix_euclidean(X_test, self.centroids)
        if self.distance_fn == "manhattan":
            distance_matrix = self.compute_distance_matrix_manhattan(X_test, self.centroids)
        cluster_assignments = self.compute_cluster_assignments(distance_matrix)
        return cluster_assignments
