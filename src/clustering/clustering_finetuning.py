
import matplotlib.pyplot as plt
from kmeans import KMeans
from kmeans_plus_plus import KMeansPlusPlus
import numpy as np

#implement elbow method from scratch
def elbow_method(model_choice, X_train, K_max):
    """
    Runs elbow method and creates a plot of K vs final SSE
    ---------
    Arguments
    ---------
    model_choice [str]: Whether to run KMeans or KMeans++
    X_train [np.array]: Numpy array of shape (m,d) denoting dataset
    K_max [int]: maximum value of K in graph
    -------
    Returns
    -------
    centroids[np.array]: Numpy array of shape (K,d) denoting centroids
    """    
    SSE_list = []
    for K in range(1,K_max+1):
        if model_choice=="KMeans":
            model = KMeans(initialization_mode="forgy", distance_fn="euclidean", num_clusters=K)
        elif model_choice=="KMeans++":
            model = KMeansPlusPlus(distance_fn="euclidean", num_clusters=K)
        else:
            raise ValueError("Invalid model choice.")
        model.train(X_train)
        SSE_list.append(model.final_SSE)

    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(1,K_max+1), SSE_list, '-bo')
    plt.xticks(np.arange(1,K_max+1))
    plt.grid()
    plt.title("Elbow Method Curve {}".format(model_choice))
    plt.xlabel('Number of clusters K')
    plt.ylabel('SSE (sum of squared errors)')
    plt.savefig("{}_elbow_curve.png".format(model_choice))


#implement silhouette method from scratch
def compute_silhouette_coefficients(model, X_train, K):
    """
    Calculates Silhouette coefficients given dataset and K
    ---------
    Arguments
    ---------
    model [<KMeans> or <KMeansPlusPlus> object]: An instance of KMeans or KMeansPlusPlus class
    X_train [np.array]: Numpy array of shape (m,d) denoting dataset
    K [int]: value of K (number of clusters)
    -------
    Returns
    -------
    silhouette_coefficients[np.array]: Numpy array of shape (m,) denoting silhouette coefficients
    """    
    distance_matrix = model.compute_distance_matrix_euclidean(X_train, model.centroids)
    closest_two_centroids = distance_matrix.argsort(1)[:,:2]
    cluster_assignments_best = closest_two_centroids[:,0]
    cluster_assignments_next_best = closest_two_centroids[:,1]

    a_vector = np.zeros(X_train.shape[0])
    b_vector = np.zeros(X_train.shape[0])

    m = len(X_train)
    for idx in range(m):
        cluster_asg = cluster_assignments_best[idx]
        cluster_nearest = cluster_assignments_next_best[idx]
        points_same_cluster = X_train[cluster_assignments_best==cluster_asg]
        points_nearest_cluster = X_train[cluster_assignments_best==cluster_nearest]
        a_value = np.linalg.norm(X_train[idx]-points_same_cluster,axis=1).mean()
        b_value = np.linalg.norm(X_train[idx]-points_nearest_cluster,axis=1).mean()
        a_vector[idx] = a_value
        b_vector[idx] = b_value
    silhouette_coefficients = (b_vector-a_vector)/np.maximum(a_vector, b_vector)
    return silhouette_coefficients


def plot_silhouette(silhouette_coefficients, cluster_assignments, K, ax):
    """
    Plots a silhouette graph on a matplotlib axis (subplot) in place
    ---------
    Arguments
    ---------
    silhouette_coefficients[np.array]: Numpy array of shape (m,) denoting silhouette coefficients
    cluster_assignments[np.array]: Numpy array of shape (m,) denoting cluster assignments
    K [int]: value of K (number of clusters)
    ax [<matplotlib.pyplot.axes> object]: An instance of matplotlib.pyplot.axes to be used for plotting
    -------
    Returns
    -------
    None
    """    
    y_ticks = []
    y_lower, y_upper = 0, 0
    for k in range(K):
        cluster_silhouette_coeffs = silhouette_coefficients[cluster_assignments == k]
        y_upper += len(cluster_silhouette_coeffs)
        ax.barh(range(y_lower, y_upper), np.sort(cluster_silhouette_coeffs), edgecolor='none', height=1)
        ax.text(-0.03, (y_lower + y_upper) / 2, str(k))
        y_lower += len(cluster_silhouette_coeffs)
    avg_silhouette_score = silhouette_coefficients.mean()
    ax.axvline(avg_silhouette_score, linestyle='--', linewidth=2, color='green')
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.set_xlabel('Silhouette coefficient values')
    ax.set_ylabel('Cluster labels')
    ax.text(.5,.9, "avg silhouette score={0:.6f}".format(avg_silhouette_score), \
            horizontalalignment="left",verticalalignment="bottom",transform=ax.transAxes)


def silhouette_method(model_choice, X_train, K_max):
    """
    Runs silhouette method for K=2 to K=K_max and obtains silhouette plots for each
    ---------
    Arguments
    ---------
    model_choice [str]: Whether to run KMeans or KMeans++
    X_train [np.array]: Numpy array of shape (m,d) denoting dataset
    K_max[int]: Max value of K for which we need to do silhouette analysis
    -------
    Returns
    -------
    None
    """    
    plt.figure(figsize=(10,6*K_max))
    for K in range(2,K_max+1):
        if model_choice=="KMeans":
            model = KMeans(initialization_mode="forgy", distance_fn="euclidean", num_clusters=K)
        elif model_choice=="KMeans++":
            model = KMeansPlusPlus(distance_fn="euclidean", num_clusters=K)
        else:
            raise ValueError("Invalid model choice.")
        model.train(X_train)
        silhouette_coefficients = compute_silhouette_coefficients(model, X_train, K)
        
        # Silhouette plot
        ax = plt.subplot(K_max-1, 1, K-1)
        plot_silhouette(silhouette_coefficients, model.cluster_assignments, K, ax)
        ax.set_title('Silhouette plot for {}: $K$={}'.format(model_choice,K), y=1.02)
    plt.savefig("{}_silhoutte_curve.png".format(model_choice))
