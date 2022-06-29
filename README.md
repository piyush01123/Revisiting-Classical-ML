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

## Installation
```
pip install RCML
```

## Import classical ML algorithm implementations
```
# Classification models
from RCML import KNN_Classifier
from RCML import Decision_Tree
from RCML import Logistic_Regression
from RCML import SVM
from RCML import Naive_Bayes

# Clustering models
from RCML import KMeans
from RCML import KMeansPlusPlus

# Regression models
from RCML import KNN_Regressor 
from RCML import Linear_Regressor

# Sequence models
from RCML import viterbi as HMM
```

See examples of usage in the repo in files having prefix `run_*`
