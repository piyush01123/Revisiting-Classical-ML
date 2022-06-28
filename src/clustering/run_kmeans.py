

import utils
from clustering_finetuning import elbow_method, silhouette_method
from clustering_utils import predict_class_from_clustering, plot_TSNE_with_clustering
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import accuracy_score, f1_score
from kmeans import KMeans
from kmeans_plus_plus import KMeansPlusPlus
import pandas as pd
from sklearn.model_selection import train_test_split


def run_kmeans_expts(X_train,y_train,X_val,y_val):
    X_train_subset = X_train[:1000]
    print("Running Elbow method for KMeans")
    elbow_method("KMeans", X_train_subset, 15)
    print("Running Silhoutte method for KMeans")
    silhouette_method("KMeans", X_train_subset, 15)
    print("Running Elbow method for KMeans++")
    elbow_method("KMeans++", X_train_subset, 15)
    print("Running Silhoutte method for KMeans++")
    silhouette_method("KMeans++", X_train_subset, 15)

    print("Starting...")
    scores_dict = {"Initialization": [], "Distance": [], "Train SSE": [], \
                "Train homogeneity score": [], "Val homogeneity score": [], \
                "Train Accuracy": [], "Train F1 Score": [], "Val Accuracy": [], \
                "Val F1 Score": []
                }

    for init_mode in ["forgy", "random_partition", "kmeans++"]:
        for dist_fn in ["euclidean", "manhattan"]:
            print("Running with initialization =", init_mode, " and distance func =", dist_fn)
            scores_dict["Initialization"].append(init_mode)
            scores_dict["Distance"].append(dist_fn)
            if init_mode =="kmeans++":
                model = KMeansPlusPlus(distance_fn=dist_fn, num_clusters=12)
            else:
                model = KMeans(initialization_mode=init_mode, distance_fn=dist_fn, num_clusters=12)
            model.train(X_train)
            scores_dict["Train SSE"].append(model.final_SSE)
            train_score = homogeneity_score(labels_true=y_train.ravel(), labels_pred=model.cluster_assignments)
            scores_dict["Train homogeneity score"].append(train_score)
            preds = predict_class_from_clustering(X_train_cluster_assignments=model.cluster_assignments, y_train=y_train,\
                    X_cluster_assignments=model.cluster_assignments,n_clusters=12, n_classes=10)
            scores_dict["Train Accuracy"].append(accuracy_score(y_train, preds))
            scores_dict["Train F1 Score"].append(f1_score(y_train, preds, average="macro"))

            val_set_cluster_assignments=model.predict(X_val)
            preds = predict_class_from_clustering(X_train_cluster_assignments=model.cluster_assignments, y_train=y_train,\
                    X_cluster_assignments=val_set_cluster_assignments,n_clusters=12, n_classes=10)
            val_score = homogeneity_score(labels_true=y_val.ravel(), labels_pred=val_set_cluster_assignments)
            scores_dict["Val homogeneity score"].append(val_score)
            scores_dict["Val Accuracy"].append(accuracy_score(y_val, preds))
            scores_dict["Val F1 Score"].append(f1_score(y_val, preds, average="macro"))
            plot_TSNE_with_clustering(X_val, 12, val_set_cluster_assignments, "tsne_{}_{}.png".format(init_mode,dist_fn), num_samples_to_use=1000)

    pd.DataFrame(scores_dict).to_csv("kmeans_result.csv", index=False)


def main():
    X_train, y_train = utils.load_mnist(path="data", kind="train")
    X_val, y_val = utils.load_mnist(path="data", kind="val")
    print("Shapes of Train X/Y", X_train.shape, y_train.shape)
    print("Shapes of Val X/Y", X_val.shape, y_val.shape)
    run_kmeans_expts(X_train,y_train,X_val,y_val)

    X_data, y_data = utils.read_latent_representation("data/data.csv")
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    X_train, X_val = utils.flatten_and_normalize_data(X_train, X_val, ndims=192)
    print("Shapes of Train X/Y", X_train.shape, y_train.shape)
    print("Shapes of Val X/Y", X_val.shape, y_val.shape)
    run_kmeans_expts(X_train,y_train,X_val,y_val)

if __name__=="__main__":
    main()