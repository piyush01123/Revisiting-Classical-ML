
import numpy as np
import matplotlib.pyplot as plt


def plot_TSNE_with_clustering(X_test, K, test_set_cluster_assignments, num_samples_to_use=1000):
    """
    Plots t-SNE graph with color according to cluster assignments
    ---------
    Arguments
    ---------
    X_test [np.array]: Numpy array of shape (m,d) denoting test dataset
    K [int]: K value
    test_set_cluster_assignments[np.array]: Numpy array of shape (m',) denoting
                                            cluster assignments  of test set
    num_samples_to_use[int]: Number of samples that we intend to use for t-SNE graph
    -------
    Returns
    -------
    None
    """
    t1 = time.time()
    X_embedded = TSNE(n_components=2).fit_transform(X_test[:num_samples_to_use])
    t2 = time.time()
    print("Time taken for t-SNE:{} sec".format(t2-t1))
    plt.figure(figsize=(10,8))
    for k in range(K):
        plt.scatter(X_embedded[test_set_cluster_assignments[:num_samples_to_use]==k,0], \
                    X_embedded[test_set_cluster_assignments[:num_samples_to_use]==k,1],\
                    label="cluster {}".format(k))
    plt.legend()

def predict_class_from_clustering(X_train_cluster_assignments, y_train, X_cluster_assignments, \
                                  n_clusters, n_classes):
    """
    Predicts class given train set cluster assignments, train set labels and a new
    set of cluster assignments. It works by assigning a class to each cluster.
    This assignment is done by voting of cluster members of train set. 
    ---------
    Arguments
    ---------
    X_train_cluster_assignments [np.array]: Numpy array of shape (m,) denoting train
                                            set cluster assignments
    y_train [np.array]: Numpy array of shape (m,) denoting train set ground truth labels
    X_cluster_assignments [np.array]: Numpy array of shape (m',) denoting new set of 
                                      cluster assignments on which prediction is to be done
    n_clusters [int]: Number of clusters
    n_classes [int]: Number of classes in ground truth
    -------
    Returns
    -------
    y_preds [np.array]: Numpy array of shape (m',) denoting predicted labels
    """
    cluster_classes = np.zeros((n_clusters,))
    for k in range(n_clusters):
        voting = np.bincount(y_train[X_train_cluster_assignments==k], minlength=n_classes)
        cluster_classes[k] = voting.argmax()
    m = len(X_cluster_assignments)
    y_preds = np.zeros((m,))
    for k in range(n_clusters):
        y_preds[X_cluster_assignments==k] = cluster_classes[k]
    return y_preds
