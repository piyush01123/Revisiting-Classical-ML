
import utils
from sklearn.model_selection import train_test_split
from split_metrics import entropy_metric, gini_metric, misc_rate_metric
from decision_tree import Decision_Tree, preOrder, sort_features_by_predictive_value
import pandas as pd


def run_dtree_regression_expts(X_train,y_train,X_val,y_val):
    dtree = Decision_Tree(max_depth=100,metric=entropy_metric)
    dtree.build_tree(X_train, y_train)
    y_hat = dtree.predict(X_val)
    acc = sum(y_hat==y_val)/len(y_val)
    print("Metric: Entropy", "Accuracy: ", acc)

    dtree = Decision_Tree(max_depth=100,metric=gini_metric)
    dtree.build_tree(X_train, y_train)
    y_hat = dtree.predict(X_val)
    acc = sum(y_hat==y_val)/len(y_val)
    print("Metric: Gini Index", "Accuracy: ", acc)

    dtree = Decision_Tree(max_depth=100,metric=misc_rate_metric)
    dtree.build_tree(X_train, y_train)
    y_hat = dtree.predict(X_val)
    acc = sum(y_hat==y_val)/len(y_val)
    print("Metric: Mis-classification rate", "Accuracy: ", acc)

    dictionary = {"metric":[],"max_depth":[],"min_size_node":[],"accuracy":[]}
    for metric, metric_fn in zip(["Entropy","Gini Index", "Misclassification Rate"],\
                                [entropy_metric, gini_metric, misc_rate_metric]):
        for md in [100,10,5,4,3]:
            for msn in [3,5,10,20]:
                dtree = Decision_Tree(max_depth=md,metric=metric_fn,min_size_node=msn)
                dtree.build_tree(X_train, y_train)
                y_hat = dtree.predict(X_val)
                acc = sum(y_hat==y_val)/len(y_val)
                print("Metric: {}, Max Depth: {}, Min Size Node: {}, Accuracy: {}".format(\
                            metric, md, msn, acc))
                dictionary["metric"].append(metric)
                dictionary["max_depth"].append(md)
                dictionary["min_size_node"].append(msn)
                dictionary["accuracy"].append(acc)

    pd.DataFrame(dictionary).to_csv("dtree_results.csv", index=False)

    print("Exporting best tree")
    dtree = Decision_Tree(max_depth=3,metric=misc_rate_metric)
    dtree.build_tree(X_train, y_train)
    y_hat = dtree.predict(X_val)
    acc = sum(y_hat==y_val)/len(y_val)
    print("Best model accuracy: ", acc)
    f = open("best_tree_preorder.txt",'w')
    f.write('\n'.join(preOrder(dtree.root,string=[])))
    f.close()

    print("Decision Tree with feature selection using chi2")
    num_features = 3
    sorted_cols = sort_features_by_predictive_value(X_train,y_train)
    X_train = X_train[sorted_cols[:num_features]]
    X_val = X_val[sorted_cols[:num_features]]
    for metric, metric_fn in zip(["Entropy","Gini Index", "Misclassification Rate"],\
                                [entropy_metric, gini_metric, misc_rate_metric]):
        dtree = Decision_Tree(max_depth=3,metric=metric_fn)
        dtree.build_tree(X_train, y_train)
        y_hat = dtree.predict(X_val)
        acc = sum(y_hat==y_val)/len(y_val)
        print("Metric:", metric, "Accuracy: ", acc)


def main():
    X_data, y_data = utils.prepare_WBCD_dataset("breast-cancer-wisconsin.data")
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    print("Shapes of Train X/Y", X_train.shape, y_train.shape)
    print("Shapes of Val X/Y", X_val.shape, y_val.shape)
    run_dtree_regression_expts(X_train,y_train,X_val,y_val)

if __name__=="__main__":
    main()