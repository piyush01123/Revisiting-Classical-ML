
import numpy as np


def entropy(p):
    """
    Calculates entropy given probability
    
    Arguments
    ---------
    p (float): probability of sample being 0 or 1 [answer is same in both cases]

    Returns
    --------
    float: entropy value
    """
    if p<1e-6 or p>1-1e-6:
        return 0
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))


def gini(p):
    """
    Calculates gini index given probability
    
    Arguments
    ---------
    p (float): probability of sample being 0 or 1 [answer is same in both cases]

    Returns
    --------
    float: Gini index
    """
    return 2*p*(1-p) # For binary class


def misc_rate(p):
    """
    Calculates misclassification rate given probability
    
    Arguments
    ---------
    p (float): probability of sample being 0 or 1 [answer is same in both cases]

    Returns
    --------
    float: Misclassification rate
    """
    return 1-max(p,1-p)


def entropy_metric(target_col):
    """
    Calculates entropy of a  pandas column
    
    Arguments
    ---------
    target_col (pd.Series): A column of a pandas dataframe

    Returns
    --------
    float: entropy value
    """
    n = len(target_col)
    p = 1/n*sum(target_col)    
    return entropy(p)


def gini_metric(target_col):
    """
    Calculates gini index of a  pandas column
    
    Arguments
    ---------
    target_col (pd.Series): A column of a pandas dataframe

    Returns
    --------
    float: gini index
    """
    n = len(target_col)
    p = 1/n*sum(target_col)    
    return gini(p)


def misc_rate_metric(target_col):
    """
    Calculates misclassification rate of a  pandas column
    
    Arguments
    ---------
    target_col (pd.Series): A column of a pandas dataframe

    Returns
    --------
    float: misclassification rate
    """
    n = len(target_col)
    p = 1/n*sum(target_col)    
    return misc_rate(p)


def verify_metrics():
    """A utility function to verify that the function return above are correct"""
    plt.figure(figsize=(10,8))
    xs = np.arange(.01,1,.01)
    ents = [entropy(x) for x in xs]
    ginis = [gini(x) for x in xs]
    mcr = [misc_rate(x) for x in xs]
    plt.plot(xs,ents,label='Entropy')
    plt.plot(xs,ginis,label='Gini Index')
    plt.plot(xs,mcr,label='Misclassification Rate')
    plt.legend()
    plt.xlabel("P(Y=1)")
    plt.ylabel("Metric")
    plt.title("Probability vs metrics used for splitting a node in DTree")
    plt.savefig("split_metrics.png")
    

if __name__=="__main__":
    import matplotlib.pyplot as plt
    verify_metrics()