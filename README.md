


# Revisiting Classsical ML
Implementations of classical ML algorithms in Numpy. Algorithms covered:
- Linear Regression
- Logistic Regression
- Support Vector Machines
- K Nearest Neighbours (both classifier and regressor)
- Naive Bayes
- K Means Clustering
- Decision Trees
- HMM

# Installation
```
pip install -r requirements.txt
```

# Import classical ML algorithm implementations
```
from classification.knn.knn_classification import KNN_Classifier
from classification.dtree.decision_tree import Decision_Tree
from classification.log_reg.logistic_regression import LogisticRegression
from classification.svm.svm import solve_hard_SVM as SVM
from classification.nv_bayes.naive_bayes import NaiveBayes

from clustering.kmeans import KMeans
from clustering.kmeans_plus_plus import KMeansPlusPlus

from regression.knn_regression import KNN_Regressor
from regression.linear_regression import LinearRegression

from sequence import viterbi
```

See examples of usage in the repo in files having prefix `run_*`