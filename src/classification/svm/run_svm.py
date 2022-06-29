
import numpy as np
from svm import solve_hard_SVM, plot_solution,get_QP_parameters_hard_SVM
import matplotlib.pyplot as plt

def main():
    # Create Synthetic Data and visualize the points
    X_data = np.array([[-3.5, -1], [-3, 0], [-3, 1], [-2.7, -1.3], [-2, -1], [-2, -2.7], 
                [-1, -2.5], [0, -3], [-1.1, 0], [0, 2.5], [1, 2], [0.7, 4], 
                [2.1, 0.2], [2.3, 1], [2.8, 1.8], [2.2, 2.8]])
    y_data = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1])

    alpha, w, b = solve_hard_SVM(X_data,y_data)
    print("alphas: {} \nw: {} \nb: {}".format(alpha.tolist(), w, b))

    plt.figure(figsize=(10,8))
    ax = plt.gca()
    plot_solution(X_data,y_data,w,b,ax)
    ax.set_title("All points")
    plt.savefig("all_points.png")

    # Write your code here
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    # plot_solution(X[alpha>0],y[alpha>0],w,b,ax,plot_w_vector=True)
    is_sv = np.where(y_data*(X_data@w+b)<=1+1e-6)
    plot_solution(X_data[is_sv],y_data[is_sv],w,b,ax,True)
    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)
    ax.set_title("Support vectors only")
    plt.savefig("supp_vec_only.png")

if __name__=="__main__":
    main()