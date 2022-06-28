
import numpy as np
import pandas as pd


class Node:
    """
    Decision Tree Node
    """
    def __init__(self, name, isLeaf, base_ent=0, depth=1, splitVal=None, left=None, right=None):
        """
        Constructor method for decision tree node
        Arguments
        ---------
        name (str): Name of the node; if the node is a terminal node then, it is the class label of the 
                    majority class at that node otherwise it is the name of the attribute being used for splitting
        isLeaf(bool): Is the node a terminal node?
        splitVal(int/str/object): The value of the feature used for splitting the node. 
                                 (We are treating everything as categorical)
        left (Node): The left child node object
        right (Node): The right child node object
        """
        self.name = name # attribute if not isLeaf else class name
        self.isLeaf = isLeaf # True if leaf node false otherwise
        self.splitVal = splitVal # if attribute==splitVal go left else right; NULL if terminal node
        self.left = left
        self.right = right
        self.depth = depth

def get_optimum_split(x_train, y_train, metric):
    """
    Finds out the optimal split given a training dataset
    Arguments
    ---------
    x_train (pd.DataFrame): Training data X
    y_train (pd.Series): Training data Y
    metric (<python method>): The function used for splitting. Can be either entropy_metric 
                              or gini_metric or misc_rate_metric
    Returns
    -------
    str: The optimal feature name
    str/int/object: The value of the feature used for splitting [We are using everything as categorical]
    """
    n = len(x_train)
    cols = []
    vals = []
    ents = []
    for col in x_train.columns:
        for val in sorted(x_train[col].unique()):
            p = sum(x_train[col]==val)/n
            e1 = metric(y_train[x_train[col]==val])
            if p == 1.0:
                e2 = 0
            else:
                e2 = metric(y_train[x_train[col]!=val])
            ent = p*e1+(1-p)*e2
            cols.append(col)
            vals.append(val)
            ents.append(ent)
#             print(col,val,ent)
    idx = np.array(ents).argsort()[0]
    return cols[idx], vals[idx]

def split_df(x_train, y_train, attr, attr_val):
    """
    Splits a training dataset into 2 parts; 1 for the left child and another for the right child
    Note: Considers the feature as categorical.
    In case the features were not categorical; we could replace the == and != by <= and > respectively
    or we could write a logic for finding the optimal split better than the naive linear search

    Arguments
    ---------
    x_train (pd.DataFrame): Training data X
    y_train (pd.Series): Training data Y
    attr (str): The feature being used to split
    attr_val (str/int/object): The value of the feature used for splitting [We are using 
                               everything as categorical]
    Returns
    -------
    pd.DataFrame: X for the left child node
    pd.Series: Y for the left child node
    pd.DataFrame: X for the right child node
    pd.Series: Y for the right child node
    """
    left_x = x_train[x_train[attr]==attr_val]
    left_y = y_train[x_train[attr]==attr_val]
    right_x = x_train[x_train[attr]!=attr_val]
    right_y = y_train[x_train[attr]!=attr_val]
    cols = x_train.columns.tolist()
    cols.remove(attr)
    left_x = left_x[cols]
    right_x = right_x[cols]
    return left_x, left_y, right_x, right_y

class Decision_Tree():
    """    Decision Tree implementation    """
    def __init__(self, metric, max_depth=5, min_size_node=3):
        """
        Constructor for the Decision tree object
        Arguments
        ---------
        max_depth (int): Max allowed depth of the decision tree. This can be tuned for validation
        metric (<python method>): The function used for splitting during decision tree building. 
                                  Can be either entropy_metric or gini_metric or misc_rate_metric
                                  for entropy, gini index and misclassification rate respectively
        min_size_node (int): The minimum number of samples a node must have to make it eligible for
                              the splitting
        """
        self.max_depth = max_depth
        self.metric = metric
        self.min_size_node = min_size_node
        self.root = None
    
    def build_tree(self, x_train, y_train, depth=1):
        """
        Builds the decision tree given training data
        Arguments
        ---------
        x_train (pd.DataFrame): Training data X
        y_train (pd.Series): Training data Y
        depth (int): Depth at which the method has been called
        
        """
        if len(y_train)==0 or len(x_train.columns)==0:
            random_label = "Malignant" # tried both, this gives slightly better result
            return Node(name=random_label, isLeaf=True)
        if depth==1:
            # Called from outside. Recursion starts from here
            opt_attr, opt_attr_val = get_optimum_split(x_train, y_train, self.metric)
            x_left, y_left, x_right, y_right = split_df(x_train, y_train, opt_attr, opt_attr_val)
            left = self.build_tree(x_left, y_left, depth+1)
            right = self.build_tree(x_right, y_right, depth+1)
            self.root = Node(name=opt_attr, isLeaf=False, depth=1, \
                             splitVal=opt_attr_val, left=left, right=right
                            )
            return
        elif depth>=self.max_depth or len(x_train)<=self.min_size_node or self.metric(y_train)<1e-6:
            p = sum(y_train)/len(y_train)
#             print(p, depth, x_train.shape, 'Terminal')
            name = 'Benign' if p<0.5 else "Malignant"
            return Node(name=name, isLeaf=True, depth=depth)
        else:
#             print(x_train.shape, 'NonTerminal')
            opt_attr, opt_attr_val = get_optimum_split(x_train, y_train, self.metric)
            x_left, y_left, x_right, y_right = split_df(x_train, y_train, opt_attr, opt_attr_val)
            left = self.build_tree(x_left, y_left, depth+1)
            right = self.build_tree(x_right, y_right, depth+1)            
            return Node(name=opt_attr, isLeaf=False, depth=depth, \
                        splitVal=opt_attr_val, left=left, right=right
                       )

    def predict_sample(self, x):
        """Traverse the tree for prediction on a single sample"""
        curr = self.root
        while not curr.isLeaf:
            if x[curr.name]==curr.splitVal:
                curr = curr.left
            else:
                curr = curr.right
        return curr.name
    
    def predict(self, x_test):
        """
        For each sample call the predict_sample method and return predictions
        Arguments
        ---------
        x_test (pd.DataFrame): Test X
        Returns
        -------
        np.array: Predictions
        """
        y_hat = [None]*len(x_test)
        ctr = 0
        for i in x_test.index:
            x = x_test.loc[i]
            y_hat[ctr] = self.predict_sample(x)
            ctr += 1
        y_hat = np.array([0 if i=="Benign" else 1 for i in y_hat])
        return y_hat

