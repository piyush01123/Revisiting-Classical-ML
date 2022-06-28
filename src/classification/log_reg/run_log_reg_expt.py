
from multi_class_utils import train_one_vs_one, predict_one_vs_one, train_one_vs_all, predict_one_vs_all
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import utils


def run_log_reg_expts(X_train,y_train,X_val,y_val):
    dictionary = {"multi_class_clf":[], "precision":[], "recall":[], "f1_score":[], "accuracy":[]}
    y_val = y_val.argmax(1)

    print("Running One vs One...")
    models_dict = train_one_vs_one(X_train,y_train)
    preds = predict_one_vs_one(X_val, models_dict,n_classes=10)
    precision = precision_score(y_val, preds, average="macro")
    recall = recall_score(y_val, preds, average="macro")
    f1_value = f1_score(y_val, preds, average="macro")
    accuracy = accuracy_score(y_val, preds)
    dictionary["multi_class_clf"].append("one vs one")
    dictionary["precision"].append(precision)
    dictionary["recall"].append(recall)
    dictionary["f1_score"].append(f1_value)
    dictionary["accuracy"].append(accuracy)

    print("Running One vs All...")
    models_dict = train_one_vs_all(X_train,y_train,n_classes=10)
    preds = predict_one_vs_all(X_val, models_dict,n_classes=10)
    precision = precision_score(y_val, preds, average="macro")
    recall = recall_score(y_val, preds, average="macro")
    f1_value = f1_score(y_val, preds, average="macro")
    accuracy = accuracy_score(y_val, preds)
    dictionary["multi_class_clf"].append("one vs all")
    dictionary["precision"].append(precision)
    dictionary["recall"].append(recall)
    dictionary["f1_score"].append(f1_value)
    dictionary["accuracy"].append(accuracy)

    pd.DataFrame(dictionary).to_csv("logistic_regression_results.csv", index=False)


def main():
    X_data, y_data = np.load("data/X.npy"), np.load("data/Y.npy")
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.30, random_state=42)
    X_train, X_val = utils.flatten_and_normalize_data(X_train, X_val, 64*64, "min_max")
    print("Shapes of Train X/Y", X_train.shape, y_train.shape)
    print("Shapes of Val X/Y", X_val.shape, y_val.shape)
    run_log_reg_expts(X_train,y_train,X_val,y_val)

if __name__=="__main__":
    main()