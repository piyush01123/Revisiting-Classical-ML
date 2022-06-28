

import utils
from clustering_finetuning import elbow_method, silhouette_method
from sklearn.metrics.cluster import homogeneity_score


def run_kmeans_expts(X_train,y_train,X_val,y_val):
    # X_train_subset = X_train[:1000]
    # print("Running Elbow method for KMeans")
    # elbow_method("KMeans", X_train_subset, 15)
    # print("Running Silhoutte method for KMeans")
    # silhouette_method("KMeans", X_train_subset, 15)
    # print("Running Elbow method for KMeans++")
    # elbow_method("KMeans++", X_train_subset, 15)
    # print("Running Silhoutte method for KMeans++")
    # silhouette_method("KMeans++", X_train_subset, 15)

    scores_dict = {"Initialization": [], "Distance": [], "Train SSE": [], \
                "Train homogeneity score": [], "Test homogeneity score": [], \
                "Train Accuracy": [], "Train F1 Score": [], "Test Accuracy": [], \
                "Test F1 Score": []
                }
    for init_mode in ["forgy", "random_partition"]:
        for dist_fn in ["euclidean", "manhattan"]:
            model = KMeans(initialization_mode=init_mode, distance_fn=dist_fn, num_clusters=12)
            model.train(trainX)
            train_score = homogeneity_score(labels_true=trainy.ravel(), labels_pred=model.cluster_assignments)
            test_set_cluster_assignments=model.predict(testX)
            test_score = homogeneity_score(labels_true=testy.ravel(), labels_pred=test_set_cluster_assignments)
            scores_dict["Initialization"].append("Forgy")
            scores_dict["Distance"].append("Euclidean")
            scores_dict["Train SSE"].append(model.final_SSE)
            scores_dict["Train homogeneity score"].append(train_score)
            scores_dict["Test homogeneity score"].append(test_score)
            preds = predict_class_from_clustering(model.cluster_assignments, trainy, \
                                      model.cluster_assignments, 12, 10)
            precision = precision_score(trainy, preds, average="macro")
            recall = recall_score(trainy, preds, average="macro")
            f1_value = f1_score(trainy, preds, average="macro")
            accuracy = accuracy_score(trainy, preds)
            conf_mat = confusion_matrix(trainy, preds)
            scores_dict["Train Accuracy"].append(accuracy)
            scores_dict["Train F1 Score"].append(f1_value)
            preds = predict_class_from_clustering(model.cluster_assignments, trainy, \
                                                test_set_cluster_assignments, 12, 10)

            precision = precision_score(testy, preds, average="macro")
            recall = recall_score(testy, preds, average="macro")
            f1_value = f1_score(testy, preds, average="macro")
            accuracy = accuracy_score(testy, preds)
            conf_mat = confusion_matrix(testy, preds)

            print("Macro avg Precision: {}".format(precision))
            print("Macro avg Recall: {}".format(recall))
            print("Macro avg F1 Score: {}".format(f1_value))
            print("Accuracy: {}".format(accuracy))
            print("Confusion matrix: \n{}".format(conf_mat))
            scores_dict["Test Accuracy"].append(accuracy)
            scores_dict["Test F1 Score"].append(f1_value)




def main():
    X_train, y_train = utils.load_mnist(path="data", kind="train")
    X_val, y_val = utils.load_mnist(path="data", kind="val")
    print("Shapes of Train X/Y", X_train.shape, y_train.shape)
    print("Shapes of Val X/Y", X_val.shape, y_val.shape)
    run_kmeans_expts(X_train,y_train,X_val,y_val)


if __name__=="__main__":
    main()