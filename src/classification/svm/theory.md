
## Hard margin SVM:

If we consider the two lines touching the support vectors parallel to the decision boundary $w^Tx+b=0$ as $w^Tx+b=+1$ and $w^Tx+b=-1$ Then the maximum margin optimization problem can be written (from formula for distance between two parallel lines) as:

$$
\begin{aligned}
    & \max \frac {2}{ \sqrt {w^Tw} }
    \\ 
     s.t. \ & \ y_i(w^Tx_i+b) \geq 1, \hspace{10pt} i=\{1,..,m\}
\end{aligned}
$$

This can be re-written as:

$$
\begin{aligned}
    & \min \frac{1}{2} w^Tw
    \\ 
     s.t. \ & \ y_i(w^Tx_i+b) \geq 1, \hspace{10pt} i=\{1,..,m\}
\end{aligned}
$$

This is primal SVM optimization problem. To derive the dual form, we write the Lagrangian:

$$
\mathcal{L}(w,b,\alpha) = \frac{1}{2} w^Tw - \sum_{i=1}^m  \alpha_i (y_i(w^Tx_i+b) - 1)
$$

where $\alpha_i\geq 0 \forall i=\{1,..,m\}$. 

From optimization we know that

$$
d^∗ = \max_{\alpha} \min_{w,b}\mathcal{L}(w,b,\alpha) \leq \min_{w,b} \max_{\alpha} \mathcal{L}(w,b,\alpha) = p^∗.
$$

where $p*,d*$ are the primal and dual optimal respectivley. For convex problem like ours $d*=p*$. Now we want to solve the unconstrained minimization problem
$\min_{w,b} \mathcal{L}(w,b,\alpha)$ to get the dual form as $\max_{\alpha} g(\alpha)$

$$
\begin{aligned}
&\frac {\partial\mathcal{L}} {\partial w} = 0 \implies w = \sum_{i=1}^m \alpha_i y_i x_i \\
&\frac {\partial\mathcal{L}} {\partial b} = 0 \implies \sum_{i=1}^m \alpha_i y_i = 0
\end{aligned}
$$

Plugging these back into Lagrangian and writing the dual form:

$$
\begin{aligned}
    &\max_\alpha \sum_{i=1}^m \alpha_i - \frac12 \sum_{i=1}^m\sum_{j=1}^m \alpha_i\alpha_j y_i y_j x_i^Tx_j\\ 
    &s.t \sum_{i=1}^m \alpha_i y_i = 0, \alpha_i \geq 0, i=\{1,..,m\}
\end{aligned}
$$

The standard quadratic programming optimization problem is:

$$
\begin{aligned}
    & \min_x \frac{1}{2} x^TPx + q^Tx
    \\
     s.t. \ & \ Gx \leq h 
    \\
    & \ Ax = b
\end{aligned}
$$

Now we construct a $m \times m$ matrix H such that $H_{ij} = y_iy_j x_i^Tx_j$ .

Take negative sign on objective to convert max to min:

$$
\begin{aligned}
    & \min_{\alpha}  \frac{1}{2}  \alpha^T H  \alpha - 1^T \alpha
    \\
    s.t. & \ (- I_m)\alpha \leq 0 
    \\
    & \ y^T \alpha = 0 
\end{aligned}
$$

We can compare this to standard QP:

$P = H$ is a matrix of size $m \times m$ such that $H_{ij} = y_iy_j x_i^Tx_j$ 

$q = - \vec 1$ is a vector of size $m \times 1$

$G = - (I_m)$ is a diagonal matrix of -1s of size $m \times m$ (Negative identity matrix)

$h  = \vec 0$ is a vector of zeros of size $m \times 1$ 

$A = y$ is the label vector of size $m \times 1$ 

$b = 0$ is a scalar


We can feed this to CVXPY and solve the QP to get $\alpha$ which can be used to obtain $w,b$ as

$$
w = \sum_{i=1}^m \alpha_i y_i x_i\\
b = 1-w^Tx_{sv}
$$

where $x_{sv}=\{ x_i, \alpha_i>0, y_i=1 \} $ is the positive side support vector.


## Soft margin SVM:

The primal form of soft margin SVM is:

$$
\begin{aligned}
    & \min_w \frac 12 w^Tw + C\sum_{i=1}^m \xi_i
    \\ 
     s.t. \ & \ y_i(w^Tx_i+b) \geq 1 - \xi_i , xi_i \geq 0 \hspace{10pt} i=\{1,..,m\}
\end{aligned}
$$

Dual:

$$
\begin{aligned}
    &\max_\alpha \sum_{i=1}^m \alpha_i - \frac12 \sum_{i=1}^m\sum_{j=1}^m \alpha_i\alpha_j y_i y_j x_i^Tx_j\\ 
    &s.t \sum_{i=1}^m \alpha_i y_i = 0, 0 \leq \alpha_i \leq C, i=\{1,..,m\}
\end{aligned}
$$

The complete derivation for soft margin SVM dual form: https://stats.stackexchange.com/a/491011/178089

Equivalent QP from dual to feed to CVXPY:

$$
\begin{aligned}
    & \min_{\alpha}  \frac{1}{2}  \alpha^T H  \alpha - 1^T \alpha
    \\
    s.t. & \ (- I_m)\alpha \leq 0 
    \\
    & (I_m)\alpha \leq C \\
    & \ y^T \alpha = 0 
\end{aligned}
$$

$ G  = [ - I_m \vert I_{m} ]^T $ is a matrix of -1s of size $2m \times m$. First $m$ rows is negative identity and last $m$ rows is positive identity matrix

$h  = [ \vec 0_m \vert \vec C_m ] $ is a vector of zeros of size $2m \times 1$.  First $m$ entities are 0 and last $m$ entities are C.

Rest of the elements $P,q,A,b$ remain same.

