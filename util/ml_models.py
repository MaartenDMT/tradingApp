import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso,
                                  LinearRegression, LogisticRegression, Ridge,
                                  SGDClassifier, SGDRegressor)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          ExtraTreeClassifier)
from xgboost import XGBClassifier

model = None
parameters = {}


def get_model(algorithm):

    # Regression Models
    if algorithm == "Linear Regression":
        model = LinearRegression()
        parameters = {
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "n_jobs": [-1, 1]
        }
    elif algorithm == "SVR":
        model = SVR()
        parameters = {
            "C": np.logspace(-3, 3, 6),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": np.arange(2, 8),
            "gamma": ["auto", "scale"],
            "tol": np.logspace(-3, -1, 4),
            "epsilon":  np.logspace(-3, 2, 6)
        }
    elif algorithm == "Ridge Regression":
        model = Ridge()
        parameters = {
            "alpha": np.logspace(-3, 3, 6),
            "fit_intercept": [True, False],
            "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            "positive": [True, False],
            "tol": np.logspace(-3, -1, 4),
            "max_iter": np.arange(1000, 2000, 500)
        }
    elif algorithm == "Lasso Regression":
        model = Lasso()
        parameters = {
            "alpha": np.logspace(-3, 3, 6),
            "fit_intercept": [True, False],
            "precompute": [True, False],
            "positive": [True, False],
            "selection": ["cyclic", "random"],
            "tol": np.logspace(-3, -1, 4),
            "max_iter": np.arange(1000, 2000, 500)
        }
    elif algorithm == "Elastic Net Regression":
        model = ElasticNet()
        parameters = {
            "alpha": np.logspace(-3, 3, 6),
            "l1_ratio": np.arange(0.5, 5, 1.5),
            "fit_intercept": [True, False],
            "precompute": [True, False],
            "positive": [True, False],
            "selection": ["cyclic", "random"],
            "tol": np.logspace(-3, -1, 4),
            "max_iter": np.arange(1000, 2000, 500)
        }
    elif algorithm == "Decision Tree Regressor":
        model = DecisionTreeRegressor()
        parameters = {
            "criterion": ["mse", "friedman_mse", "mae"],
            "splitter": ["best", "random"],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "random_state": np.arange(1, 5, 1),
            "max_leaf_nodes": np.arange(1, 5, 1),
            "min_impurity_decrease": np.logspace(-1, 2, 4),
            "ccp_alpha": np.logspace(-3, 1, 4)
        }
    elif algorithm == "Bayesian Ridge Regression":
        model = BayesianRidge()
        parameters = {
            "n_iter": np.arange(1000, 2000, 500),
            "tol": [0.001, 0.01, 0.1],
            "alpha_1": [1e-6, 1e-5, 1e-4],
            "alpha_2": [1e-6, 1e-5, 1e-4],
            "lambda_1": [1e-6, 1e-5, 1e-4],
            "lambda_2": [1e-6, 1e-5, 1e-4],
        }
    elif algorithm == "SGD Regressor":
        model = SGDRegressor()
        parameters = {
            "loss": ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            "penalty": ["l2", "l1", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.01],
            "l1_ratio": [0.15, 0.3, 0.5],
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "eta0": [0.01, 0.1, 0.5],
            "max_iter": np.arange(1000, 2000, 500),
            "shuffle": [True, False]
        }

    # Classification Models
    elif algorithm == "Logistic Regression":
        model = LogisticRegression(multi_class='auto')
        parameters = {
            "penalty": ['l2'],
            "C":  np.logspace(-1, 2, 4),
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "max_iter": np.arange(1000, 2000, 500),
            "tol": np.logspace(-3, -1, 4),
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
            "max_iter": np.arange(500, 2000, 500)
        }
    elif algorithm == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
        parameters = {
            "max_depth": [3, 5, 10],
            "min_samples_split": [3, 5, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "criterion": ["gini", "entropy"]
        }
    elif algorithm == "Random Forest Classifier":
        model = RandomForestClassifier()
        parameters = {
            "n_estimators": np.arange(10, 310, 50),
            "max_depth": np.arange(10, 310, 50),
            "min_samples_split": np.arange(2, 11, 2),
            "min_samples_leaf": np.arange(1, 6),
            "criterion": ["gini", "entropy"]
        }
    elif algorithm == "SVC":
        model = SVC()
        parameters = {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": [2, 3, 4, 5],
            "gamma": ["auto", "scale"],
            "tol": np.logspace(-3, -1, 4),
            "cache_size": np.arange(200, 500, 100),
        }

    elif algorithm == "Isolation Forest":
        model = IsolationForest()
        parameters = {
            "n_estimators": np.arange(100, 1000, 100),
            "max_samples": ["auto"],
            "contamination": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
    elif algorithm == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
        parameters = {
            "loss": ["deviance", "exponential"],
            "learning_rate": [0.1, 0.01, 0.001],
            "n_estimators": np.arange(100, 1000, 100),
            "max_depth": [3, 4, 6, 8, 10]
        }
    elif algorithm == "Extra Tree Classifier":
        model = ExtraTreeClassifier()
        parameters = {
            "max_depth": [3, 4, 6, 8, 10],
            "min_samples_split": [3, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "criterion": ["gini", "entropy"]
        }
    elif algorithm == "XGBoost Classifier":
        model = XGBClassifier(use_label_encoder=True)
        parameters = {
            "max_depth": [3, 4, 6, 8, 10],
            "eta": [0.1, 0.2, 0.3, 0.4, 0.5],
            "objective": ["binary:logistic", "multi:softmax"],
            "num_class": [3],
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
            'radius': [10, 20, 30, 40, 50],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [30, 40, 50],
            'p': [1, 2],
        }
    elif algorithm == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
        parameters = {
            "n_neighbors": [3, 5, 11, 19],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [30, 40, 50],
            "p": [1, 2]
        }
    elif algorithm == "AdaBoost Classifier":
        model = AdaBoostClassifier()
        parameters = {
            "n_estimators": np.arange(10, 310, 50),
            "learning_rate": [0.01, 0.1, 1.0],
            "algorithm": ["SAMME", "SAMME.R"]
        }
    elif algorithm == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor()
        parameters = {
            "loss": ["ls", "lad", "huber", "quantile"],
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": np.arange(10, 310, 50),
            "subsample": [1.0, 0.9, 0.8],
            "criterion": ["friedman_mse", "mse", "mae"],
            "max_depth": [3, 5, 7],
        }
    elif algorithm == "Gaussian Process Classifier":
        model = GaussianProcessClassifier()
        parameters = {
            "kernel": ["RBF", "Matern"],
            "n_restarts_optimizer": [0, 1, 2],
            "max_iter_predict": np.arange(100, 500, 100),
        }
    elif algorithm == "Quadratic Discriminant Analysis":
        model = QuadraticDiscriminantAnalysis()
        parameters = {
            "reg_param": [0.0, 0.1, 0.2],
            "store_covariance": [True, False],
            "tol": [0.0001, 0.001, 0.01],
        }
    elif algorithm == "SGD Classifier":
        model = SGDClassifier()
        parameters = {
            "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            "penalty": ["l2", "l1", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.01],
            "l1_ratio": [0.15, 0.3, 0.5],
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "eta0": [0.01, 0.1, 0.5],
        }
    else:
        raise ValueError("Invalid algorithm: {}".format(algorithm))

    return model, parameters
