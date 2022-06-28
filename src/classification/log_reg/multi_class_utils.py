
import numpy as np
import itertools
from logistic_regression import LogisticRegression, threshold

def train_one_vs_one(X_train, y_train, **kwargs):
    """
    Trains logistic regression models for multi-class classification problem using OVO
    ---------
    Arguments
    ---------
    X_train [np.array]: Numpy array of shape (m,d) denoting train dataset X
    y_train [np.array]: Numpy array of shape (m,) denoting train dataset Y containing (0,1)
    kwargs [<keyword arguments>]: Additional arguments for Logistic Regression constructor
    -------
    Returns
    -------
    models_dict [dict]: Dictionary containing models. Keys are the 2-tuples which are combinations
                       of class ids and values are the trained model instances
    """
    # Assume One Hot
    m, n_classes = y_train.shape
    class_combs = itertools.combinations(range(n_classes),2)

    models_dict = dict()
    for comb in class_combs:
        class_1, class_2 = comb
        print("Training Model {} vs {}".format(class_1, class_2))
        idx_class_1, = np.where(y_train[:, class_1])
        idx_class_2, = np.where(y_train[:, class_2])
        X_train_sub = np.concatenate([X_train[idx_class_1], X_train[idx_class_2]], axis=0)
        y_train_sub = np.concatenate([np.zeros((len(idx_class_1),)), np.ones((len(idx_class_2),))])
        # X_train_sub, y_train_sub, idx_class_1, idx_class_2 = get_data_subset(X_test,y_test, class_1, class_2)
        print("Shapes:", X_train_sub.shape, y_train_sub.shape)
        model = LogisticRegression(**kwargs)
        model.train(X_train_sub, y_train_sub)
        models_dict[comb] = model
    return models_dict


def predict_one_vs_one(X_test, models_dict, n_classes):
    """
    Obtains prediction for multi-class classification problem on test set using 
    logistic regression models  using OVO
    ---------
    Arguments
    ---------
    X_test [np.array]: Numpy array of shape (m',d) denoting test dataset X
    models_dict [dict]: Dictionary containing models. Keys are the 2-tuples which are combinations
                       of class ids and values are the trained model instances
    n_classes [int]: Number of classes in the problem
    -------
    Returns
    -------
    y_pred [np.array]: Numpy array of shape (m',) denoting predictions
    """    
    m = len(X_test)
    class_combs = itertools.combinations(range(n_classes),2)
    y_pred_vote = np.zeros((m,n_classes))
    for comb in class_combs:
        class_1, class_2 = comb
        model = models_dict[comb]
        y_hat = model.predict(X_test)
        y_hat = threshold(y_hat)
        y_pred_vote[y_hat==0,class_1] += 1
        y_pred_vote[y_hat==1,class_2] += 1
    y_pred = y_pred_vote.argmax(1)
    return y_pred


def predict_one_vs_one_v2(X_test, models_dict, n_classes):
    """
    Obtains prediction for multi-class classification problem on test set using 
    logistic regression models using OVO
    ---------
    Arguments
    ---------
    X_test [np.array]: Numpy array of shape (m',d) denoting test dataset X
    models_dict [dict]: Dictionary containing models. Keys are the 2-tuples which are combinations
                       of class ids and values are the trained model instances
    n_classes [int]: Number of classes in the problem
    -------
    Returns
    -------
    y_pred [np.array]: Numpy array of shape (m',) denoting predictions
    """    
    class_combs = itertools.combinations(range(n_classes),2)
    y_pred_vote = np.zeros((m,n_classes))
    for comb in class_combs:
        class_1, class_2 = comb
        model = models_dict[comb]
        y_hat = model.predict(X_test)
        y_hat = threshold(y_hat)
        y_pred_vote[y_hat==0,class_1] += 1
        y_pred_vote[y_hat==1,class_2] += 1
    y_pred = y_pred_vote.argmax(1)
    return y_pred

def test_each_classifier_one_vs_one(X_test, y_test, models_dict):
    """
    Tests all logistic regression models  on test set using only samples of the 
    two classes on which model was trained. This function cannot be used for prediction. 
    It is used only for debugging.
    ---------
    Arguments
    ---------
    X_test [np.array]: Numpy array of shape (m',d) denoting test dataset X
    y_test [np.array]: Numpy array of shape (m',) denoting test dataset Y
    models_dict [dict]: Dictionary containing models. Keys are the 2-tuples which are combinations
                       of class ids and values are the trained model instances
    -------
    Returns
    -------
    acc_list [list<float>]: List containing accuracies of all models
    """    
    m, n_classes = y_test.shape
    class_combs = itertools.combinations(range(n_classes),2)
    acc_list = []
    for comb in class_combs:
        class_1, class_2 = comb
        idx_class_1, = np.where(y_test[:, class_1])
        idx_class_2, = np.where(y_test[:, class_2])
        X_test_sub = np.concatenate([X_test[idx_class_1], X_test[idx_class_2]], axis=0)
        y_test_sub = np.concatenate([np.zeros((len(idx_class_1),)), np.ones((len(idx_class_2),))])
        # X_test_sub, y_test_sub, idx_class_1, idx_class_2 = get_data_subset(X_test,y_test, class_1, class_2)
        model = models_dict[comb]
        y_hat = model.predict(X_test_sub)
        y_hat = threshold(y_hat)
        acc_list.append(accuracy_score(y_test_sub, y_hat))
    return acc_list

def train_one_vs_all(X_train, y_train,n_classes, **kwargs):
    """
    Trains logistic regression models for multi-class classification problem usig OVA
    ---------
    Arguments
    ---------
    X_train [np.array]: Numpy array of shape (m,d) denoting train dataset X
    y_train [np.array]: Numpy array of shape (m,) denoting train dataset Y containing (0,1)
    n_classes [int]: Number of classes in the problem
    kwargs [<keyword arguments>]: Additional arguments for Logistic Regression constructor
    -------
    Returns
    -------
    models_dict [dict]: Dictionary containing models. Keys are class ids and values 
                        are the trained model instances
    """
    m, n_classes = y_train.shape
    models_dict = dict()
    for class_id in range(n_classes):
        y_train_mask = y_train[:,class_id]
        print("Training Model {} vs Rest".format(class_id))
        model = LogisticRegression(**kwargs)
        model.train(X_train, y_train_mask)
        models_dict[class_id] = model
        print("Training done.")
    return models_dict

def predict_one_vs_all(X_test, models_dict, n_classes):
    """
    Obtains prediction for multi-class classification problem on test set using 
    logistic regression models using OVA
    ---------
    Arguments
    ---------
    X_test [np.array]: Numpy array of shape (m',d) denoting test dataset X
    models_dict [dict]: Dictionary containing models. Keys are class ids and values 
                        are the trained model instances
    n_classes [int]: Number of classes in the problem
    -------
    Returns
    -------
    y_pred [np.array]: Numpy array of shape (m',) denoting predictions
    """    
    m = len(X_test)
    scores_matrix = np.zeros((m,n_classes))
    for class_id in range(n_classes):
        model = models_dict[class_id]
        y_hat = model.predict(X_test)
        scores_matrix[:,class_id] = y_hat
    y_pred = scores_matrix.argmax(1)
    return y_pred
