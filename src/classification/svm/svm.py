
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def get_QP_parameters_hard_SVM(X,y):
    """
    Obtains parameters P,q,G,h,A,b for QP solver given dataset (X,y) to run hard SVM
    ---------
    Arguments
    ---------
    X [np.array]: Numpy array of shape (m,d) denoting dataset X
    y [np.array]: Numpy array of shpae (m,) denoting dataset Y containing (-1,+1)
    -------
    Returns
    -------
    P [np.array]: Numpy array of shape (m,m) for QP solver
    q [np.array]: Numpy array of shape (m,1) for QP solver
    G [np.array]: Numpy array of shape (m,m) for QP solver
    h [np.array]: Numpy array of shape (m,) for QP solver
    A [np.array]: Numpy array of shape (m,) for QP solver
    b [float]: Scalar b for QP solver
    """
    m,_ = X.shape
    X_times_y = y.reshape(-1,1)*X
    P = X_times_y@X_times_y.T
    q = -np.ones((m,1)) 
    G = -np.eye(m)
    h = np.zeros(m)
    A = y
    b = 0
    return P,q,G,h,A,b

def solve_QP(P,q,G,h,A,b):
    """
    Given QP parameters runs the solver and returns the solution
    ---------
    Arguments
    ---------
    P [np.array]: Numpy array of shape (m,m) for QP solver
    q [np.array]: Numpy array of shape (m,1) for QP solver
    G [np.array]: Numpy array of shape (m,m) for QP solver
    h [np.array]: Numpy array of shape (m,) for QP solver
    A [np.array]: Numpy array of shape (m,) for QP solver
    b [float]: Scalar b for QP solver
    -------
    Returns
    -------
    x [np.array]: Numpy array of shape (m,) denoting solution of QP
    """
    m = P.shape[0]
    x = cp.Variable(m)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     [G @ x <= h,
                      A @ x == b])
    prob.solve()
    return x.value

def solve_hard_SVM(X,y):
    """
    Given dataset (X,y), runs hard SVM and returns alpha, w, b
    ---------
    Arguments
    ---------
    X [np.array]: Numpy array of shape (m,d) denoting dataset X
    y [np.array]: Numpy array of shpae (m,) denoting dataset Y containing (-1,+1)
    -------
    Returns
    -------
    alpha [np.array]: Numpy array of shape (m,) denoting alpha from SVM dual
    w [np.array]: Numpy array of shape (d,) denoting weights
    b [float]: Scalar denoting bias
    """
    P,q,G,h,A,b = get_QP_parameters_hard_SVM(X,y)
    alpha = solve_QP(P,q,G,h,A,b)
    alpha[alpha<1e-10] = 0
    w = np.sum((y*alpha).reshape(-1,1)*X, 0)
    x_sv = X[np.bitwise_and(y==1, alpha>0)]
    x_sv = x_sv[0]  #In case of multiple +sv, we get the same b for any one
    b = 1-x_sv@w
    return alpha, w, b

def plot_solution(X,y,w,b,ax,plot_w_vector=False):
    """
    Given dataset (X,y), SVM parameters (w, b) and a matplotlib axis, creates a graph
    of scatter plot of points, decision boundary, margin lines and weight vector (optionally).
    ---------
    Arguments
    ---------
    X [np.array]: Numpy array of shape (m,d) denoting dataset X
    y [np.array]: Numpy array of shpae (m,) denoting dataset Y containing (-1,+1)
    alpha [np.array]: Numpy array of shape (m,) denoting alpha from SVM dual
    w [np.array]: Numpy array of shape (d,) denoting weights
    b [float]: Scalar denoting bias
    ax [<matplotlib.pyplot.axes> object]: Matplotlib axes instance to draw graph
    plot_w_vector [bool]: Whether to draw weight vector on graph
    -------
    Returns
    -------
    None
    """
    xs = np.linspace(X[:,0].min(), X[:,0].max(), 50)
    w1,w2 = w
    y_pos_line = (1-b-w1*xs)/w2
    y_neg_line = (-1-b-w1*xs)/w2
    y_zero_line = (0-b-w1*xs)/w2

    ax.scatter(X[:,0][y==-1], X[:,1][y==-1],c='b')
    ax.scatter(X[:,0][y==1], X[:,1][y==1],c='r')

    ax.plot(xs, y_neg_line, 'b--')
    ax.plot(xs, y_pos_line, 'r--')
    ax.plot(xs, y_zero_line, 'g')

    if plot_w_vector:
        ax.arrow(x=0,y=0,dx=1,dy=w2/w1, head_width=0.2)
