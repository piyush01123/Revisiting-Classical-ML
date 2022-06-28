
import utils
from sklearn.model_selection import train_test_split
from knn_regression import KNN_Regressor
from linear_regression import LinearRegression
from metrics import mse_error, mae_error, r_squared
import pandas as pd
import matplotlib.pyplot as plt
import time


def run_knn_regression_expts(X_train,y_train,X_val,y_val):
    print("Running KNN Regression Experiments...")
    knn_model = KNN_Regressor(X_train, y_train, normalize=False)
    Y_hat = knn_model.predict(X_val,K=5,metric='euclidean')
    print("Without normalization")
    print("MSE", mse_error(Y_hat, y_val))
    print("MAE", mae_error(Y_hat, y_val))
    print("RSQ", r_squared(Y_hat, y_val))

    t1 = time.time()
    knn_model = KNN_Regressor(X_train, y_train, normalize=True)
    Y_hat = knn_model.predict(X_val,K=5,metric='euclidean')
    t2 = time.time()
    print("With normalization")
    print("MSE", mse_error(Y_hat, y_val))
    print("MAE", mae_error(Y_hat, y_val))
    print("RSQ", r_squared(Y_hat, y_val))
    print("Time taken for KNN Regression: {} secs".format(t2-t1))

    report = {}

    print("KNN with Euclidean distance without weighting")
    Y_hat = knn_model.predict(X_val,K=5,metric='euclidean')
    mse_unwt = mse_error(Y_hat, y_val)
    mae_unwt = mae_error(Y_hat, y_val)
    rsq_unwt = r_squared(Y_hat, y_val)

    print("KNN with Euclidean distance with weighting")
    Y_hat = knn_model.predict(X_val,K=5,metric='euclidean', weighting=True)
    mse_wtd = mse_error(Y_hat, y_val)
    mae_wtd = mae_error(Y_hat, y_val)
    rsq_wtd = r_squared(Y_hat, y_val)

    report['Euclidean'] = {'unweighted': {'mse':mse_unwt, 'mae':mae_unwt, 'rsq':rsq_unwt}, \
                        'weighted': {'mse':mse_wtd, 'mae':mae_wtd, 'rsq':rsq_wtd}}

    print("KNN with Manhattan distance without weighting")
    Y_hat = knn_model.predict(X_val,K=5,metric='cityblock')
    mse_unwt = mse_error(Y_hat, y_val)
    mae_unwt = mae_error(Y_hat, y_val)
    rsq_unwt = r_squared(Y_hat, y_val)

    print("KNN with Manhattan distance with weighting")
    Y_hat = knn_model.predict(X_val,K=5,metric='cityblock', weighting=True)
    mse_wtd = mse_error(Y_hat, y_val)
    mae_wtd = mae_error(Y_hat, y_val)
    rsq_wtd = r_squared(Y_hat, y_val)

    report['Manhattan'] = {'unweighted': {'mse':mse_unwt, 'mae':mae_unwt, 'rsq':rsq_unwt}, \
                        'weighted': {'mse':mse_wtd, 'mae':mae_wtd, 'rsq':rsq_wtd}}

    print("KNN with Hamming distance without weighting")
    Y_hat = knn_model.predict(X_val,K=5,metric='hamming')
    mse_unwt = mse_error(Y_hat, y_val)
    mae_unwt = mae_error(Y_hat, y_val)
    rsq_unwt = r_squared(Y_hat, y_val)

    print("KNN with Hamming distance with weighting")
    Y_hat = knn_model.predict(X_val,K=5,metric='hamming', weighting=True)
    mse_wtd = mse_error(Y_hat, y_val)
    mae_wtd = mae_error(Y_hat, y_val)
    rsq_wtd = r_squared(Y_hat, y_val)

    report['Hamming'] = {'unweighted': {'mse':mse_unwt, 'mae':mae_unwt, 'rsq':rsq_unwt}, \
                        'weighted': {'mse':mse_wtd, 'mae':mae_wtd, 'rsq':rsq_wtd}}

    dictionary = {"Metric": ["Euclidean"]*2+["Manhattan"]*2 + ["Hamming"]*2,
                "Weighting": ["Unweighted", "Weighted"]*3,

                "MSE": [report['Euclidean']['unweighted']['mse'], report['Euclidean']['weighted']['mse'], 
                        report['Manhattan']['unweighted']['mse'], report['Manhattan']['weighted']['mse'],
                        report['Hamming']['unweighted']['mse'], report['Hamming']['weighted']['mse']],

                "MAE": [report['Euclidean']['unweighted']['mae'], report['Euclidean']['weighted']['mae'], 
                        report['Manhattan']['unweighted']['mae'], report['Manhattan']['weighted']['mae'],
                        report['Hamming']['unweighted']['mae'], report['Hamming']['weighted']['mae']],

                "RSqrd": [report['Euclidean']['unweighted']['rsq'], report['Euclidean']['weighted']['rsq'], 
                        report['Manhattan']['unweighted']['rsq'], report['Manhattan']['weighted']['rsq'],
                        report['Hamming']['unweighted']['rsq'], report['Hamming']['weighted']['rsq']]

                }

    pd.DataFrame(dictionary).to_csv("knn_regression_results.csv", index=False)

    print("KNN with Euclidean distance for various K values")
    k_values = [2,3,5,7,11,16]
    rsq_values_unwt,rsq_values_wt = [],[]
    for k_val in k_values:
        print("K=", k_val, "without weighting")
        Y_hat = knn_model.predict(X_val,K=k_val,metric='euclidean')
        rsq_values_unwt.append(r_squared(Y_hat, y_val))
        print("K=", k_val, "with weighting")
        Y_hat = knn_model.predict(X_val,K=k_val,metric='euclidean',weighting=True)
        rsq_values_wt.append(r_squared(Y_hat, y_val))

    print("Exporting some results...")
    plt.figure(figsize=(10,8))
    plt.plot(k_values,rsq_values_unwt, 'o-', label="Unweighted")
    plt.plot(k_values,rsq_values_wt, 'o-', label="Weighted")
    plt.xlabel('K values')
    plt.ylabel('R Squared')
    plt.legend()
    plt.title("KNN Regression R squared using Euclidean distance")
    plt.savefig("knn_results_k_value.png")


def run_linear_regression_expts(X_train,y_train,X_val,y_val):
    print("Running Linear Regression Experiments...")
    dictionary = {"norm_mode":[], "mse":[], "mae": [], "rsqd": []}
    print("Without normalization")
    lr_model = LinearRegression(X_train, y_train, normalize=False)
    Y_hat = lr_model.predict(X_val)
    print("MSE", mse_error(Y_hat, y_val))
    print("MAE", mae_error(Y_hat, y_val))
    print("RSQ", r_squared(Y_hat, y_val))
    dictionary["norm_mode"].append("unnormalized")
    dictionary["mse"].append(mse_error(Y_hat, y_val))
    dictionary["mae"].append(mse_error(Y_hat, y_val))
    dictionary["rsqd"].append(r_squared(Y_hat, y_val))

    print("With normalization")
    t1 = time.time()
    lr_model = LinearRegression(X_train, y_train, normalize=True)
    Y_hat = lr_model.predict(X_val)
    t2 = time.time()
    print("MSE", mse_error(Y_hat, y_val))
    print("MAE", mae_error(Y_hat, y_val))
    print("RSQ", r_squared(Y_hat, y_val))
    print("Time taken for Linear Regression: {} secs".format(t2-t1))
    dictionary["norm_mode"].append("normalized")
    dictionary["mse"].append(mse_error(Y_hat, y_val))
    dictionary["mae"].append(mse_error(Y_hat, y_val))
    dictionary["rsqd"].append(r_squared(Y_hat, y_val))

    return pd.DataFrame(dictionary)


def main():
    X_data, Y_data = utils.prepare_diamonds_dataset("diamonds.csv")
    X_train, X_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.3, random_state=42)
    print("Shapes of Train X/Y", X_train.shape, y_train.shape)
    print("Shapes of Val X/Y", X_val.shape, y_val.shape)
    run_knn_regression_expts(X_train,y_train,X_val,y_val)

    result_all = run_linear_regression_expts(X_train,y_train,X_val,y_val)

    X_data, Y_data = utils.prepare_diamonds_dataset("diamonds.csv", fewer_columns=True)
    X_train, X_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.3, random_state=42)
    print("Shapes of Train X/Y", X_train.shape, y_train.shape)
    print("Shapes of Val X/Y", X_val.shape, y_val.shape)
    result_fewer = run_linear_regression_expts(X_train,y_train,X_val,y_val)

    result_all["fewer_columns"] = False
    result_fewer["fewer_columns"] = True
    pd.concat([result_all,result_fewer]).to_csv("linear_regression_results.csv", columns=\
              ["fewer_columns","norm_mode","mse","mae","rsqd"], index=False)


if __name__ == "__main__":
    main()