
from naive_bayes import NaiveBayes
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def main():
    digits = datasets.load_digits()
    X_train, X_val, y_train, y_val = train_test_split(digits.images, digits.target, test_size=.2)
    nb_model = NaiveBayes(10)
    nb_model.train(X_train,y_train)
    pred = nb_model.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, pred))
    print("F1 Score:", f1_score(y_val, pred, average="macro"))

if __name__=="__main__":
    main()