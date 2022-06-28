
import utils
from knn_classification import KNN_Classifier
import numpy as np
import matplotlib.pyplot as plt
import time


def run_knn_classification_expts(X_train, y_train, X_val, y_val):
    print("Running KNN Classification Experiments...")
    classifier = KNN_Classifier()
    classifier.train(X_train, y_train)

    times = {"num_loops":[], "time_in_secs":[]}
    t1 = time.time()
    dists = classifier.compute_distances_two_loops(X_val)
    t2 = time.time()
    times["num_loops"].append(2)
    times['time_in_secs'].append(t2-t1)
    t1 = time.time()
    dists = classifier.compute_distances_one_loop(X_val)
    t2 = time.time()
    times["num_loops"].append(1)
    times['time_in_secs'].append(t2-t1)
    t1 = time.time()
    dists = classifier.compute_distances_no_loop(X_val)
    t2 = time.time()
    times["num_loops"].append(0)
    times['time_in_secs'].append(t2-t1)
    times.to_csv("d_mat_cal_times.csv", index=False)

    accuracies = {}
    k_values = [1,3,5,7]
    for k_val in k_values:
        y_val_pred = classifier.predict_labels(dists, k_val)
        num_correct = np.sum(y_val_pred == y_val)
        accuracy = float(num_correct) / len(y_val)
        accuracies[k_val] = accuracy
    plt.figure(figsize=(14,5))
    plt.plot(k_values, [accuracies[k] for k in k_values], 'o-',label='acc')
    plt.xlabel('K')
    plt.ylabel('Acc')
    plt.title('Acc vs K for KNN classification')
    plt.legend()
    plt.savefig("knn_classifier_result.png")


def main():
    X_train, y_train, X_val, y_val= utils.prepare_cifar_dataset("cifar-10-batches-py")
    print("Shapes of Train X/Y", X_train.shape, y_train.shape)
    print("Shapes of Val X/Y", X_val.shape, y_val.shape)
    run_knn_classification_expts(X_train,y_train,X_val,y_val)

if __name__ == "__main__":
    main()