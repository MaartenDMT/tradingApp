from sklearn.ensemble import (GradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier

model = None
parameters = {}


def get_model(algorithm):
    if algorithm == "Linear Regression":
        model = LinearRegression()
        parameters = {
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "n_jobs": [-1, 1]
        }
    elif algorithm == "Logistic Regression":
        model = LogisticRegression(multi_class='auto')
        parameters = {
            "penalty": ['l2'],
            "C": [0.1, 1.0, 10.0],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "max_iter": [1000, 1500, 2000]
        }
    elif algorithm == "MLPClassifier":
        model = MLPClassifier()
        parameters = {
            "activation": ['relu', 'identity', 'logistic', 'tanh'],
            "hidden_layer_sizes": [(10,), (20,), (10, 10)],
            "alpha": [0.0001, 0.0002, 0.001],
            "solver": ["sgd", "adam"],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "momentum": [0.9, 0.95, 0.99],
            "max_iter": [200, 500, 1_000]
        }
    elif algorithm == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
        parameters = {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "criterion": ["gini", "entropy"]
        }
    elif algorithm == "Random Forest Classifier":
        model = RandomForestClassifier()
        parameters = {
            "n_estimators": [10, 50, 100, 200, 300],
            "max_depth": [2, 4, 6, 8, 10],
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "criterion": ["gini", "entropy"]
        }
    elif algorithm == "SVC":
        model = SVC()
        parameters = {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": [2, 3, 4, 5],
            "gamma": ["auto", "scale"]
        }
    elif algorithm == "SVR":
        model = SVR()
        parameters = {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": [2, 3, 4, 5],
            "gamma": ["auto", "scale"]
        }
    elif algorithm == "Isolation Forest":
        model = IsolationForest()
        parameters = {
            "n_estimators": [10, 50, 100, 200, 300],
            "max_samples": ["auto"],
            "contamination": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
    elif algorithm == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
        parameters = {
            "loss": ["deviance", "exponential"],
            "learning_rate": [0.1, 0.01, 0.001],
            "n_estimators": [10, 50, 100, 200, 300],
            "max_depth": [2, 4, 6, 8, 10]
        }
    elif algorithm == "Extra Tree Classifier":
        model = ExtraTreeClassifier()
        parameters = {
            "max_depth": [2, 4, 6, 8, 10],
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "criterion": ["gini", "entropy"]
        }
    elif algorithm == "XGBoost Classifier":
        model = XGBClassifier()
        parameters = {
            "max_depth": [2, 4, 6, 8, 10],
            "eta": [0.1, 0.2, 0.3, 0.4, 0.5],
            "objective": ["binary:logistic", "multi:softmax"],
            "num_class": [1, 2, 3, 4],
            "eval_metric": ["mlogloss"]
        }
    # TODO: refactor the errors
    elif algorithm == "Gaussian Naive Bayes":
        model = GaussianNB()
        parameters = {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    elif algorithm == "Radius Neighbors Classifier":
        model = RadiusNeighborsClassifier()
        parameters = {
            'radius': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [30, 40, 50],
            'p': [1, 2],
            'outlier_label': ['outliners']
        }
    else:
        raise ValueError("Invalid algorithm: {}".format(algorithm))

    return model, parameters
