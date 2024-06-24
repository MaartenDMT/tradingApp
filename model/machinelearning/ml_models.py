import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              GradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier)
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (BayesianRidge, ElasticNet, HuberRegressor,
                                  Lars, Lasso, LassoCV, LinearRegression,
                                  LogisticRegression, RANSACRegressor, Ridge,
                                  SGDClassifier, SGDRegressor,
                                  TheilSenRegressor)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier

model = None
parameters = {}

max_iter = np.arange(2000, 10_000, 2000)
n_estimators = np.arange(1000, 5000, 1000)
learning_rate = np.logspace(-3, 0, 4)
tol = np.logspace(-3, -1, 3)
alpha = np.logspace(-3, 0, 4)


def get_model(algorithm):
    """
    Get the model for the training.

    :param algorithm: the name of the model to be selected.
    :return: SKlearn Model.
    """

    # Regression Models
    if algorithm == "Linear Regression":
        model = LinearRegression()
        parameters = {
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "n_jobs": [-1]
        }
    elif algorithm == "Ridge Regression":
        model = Ridge()

        # Define parameters for Ridge Regression
        parameters = {
            "alpha": [0.001],
            "fit_intercept": [True],
            "positive": [False],
            "tol": [0.1, 0.01],
            "max_iter": [6000]
        }

        # Add 'solver' to parameters based on the value of 'fit_intercept'
        solver_options = ["sag", "saga"]
        solver = solver_options if True in parameters["fit_intercept"] else [
            "sag", "saga"]
        parameters["solver"] = solver

    elif algorithm == "Lasso Regression":
        model = Lasso()

        # Initially define the parameters for Lasso Regression
        parameters = {
            "alpha": [1, 0.5],
            "fit_intercept": [True],
            "precompute": [True, False],
            "positive": [False],
            "tol": [0.001],
            "max_iter": [4000, 5000]
        }

        # Conditionally modify the 'selection' parameter based on the value of 'fit_intercept'
        if True in parameters["fit_intercept"]:
            selection = ["cyclic", "random"]
        else:
            selection = None
        parameters["selection"] = selection

    elif algorithm == "Elastic Net Regression":
        model = ElasticNet()
        parameters = {
            "alpha": [0.1, 0.01],
            "l1_ratio": [0.7, 0.6],
            "fit_intercept": [True],
            "precompute": [True],
            "positive": [True, False],
            "selection":  "random",
            "tol": [0.1],
            "max_iter": [4000, 5000]
        }
    elif algorithm == "Bayesian Ridge Regression":
        model = BayesianRidge()
        parameters = {
            "n_iter": n_estimators,
            "max_iter": [6000, 7000],
            "tol": [0.0016362436332760396],
            "alpha_1":  [0.00031622776601683794],
            "alpha_2": [0.000001, 0.0001],
            "lambda_1": [0.000001, 0.0001],
            "lambda_2": [0.000001, 0.0001],
        }
    elif algorithm == "SVR":
        model = SVR()
        parameters = {
            "C": np.logspace(-3, 1, 4),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": np.arange(2, 8),
            "gamma": ["auto", "scale"],
            "tol": tol,
            "epsilon":  tol,
            "shrinking": [True, False]
        }
    elif algorithm == "NuSVR":
        model = NuSVR()
        parameters = {
            "nu": [0.25, 0.5, 0.75],
            "C": np.logspace(-3, 1, 4),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": np.arange(2, 8),
            "gamma": ["auto", "scale"],
            "tol": tol,
            "epsilon": tol,
            "shrinking": [True, False]
        }

    elif algorithm == "LinearSVR":
        model = LinearSVR()
        parameters = {
            "C": np.logspace(-3, 1, 4),
            "epsilon": tol,
            "tol": tol,
            "fit_intercept": [True, False],
            "max_iter": [1000, 2000, 3000],
        }
    elif algorithm == "SGD Regressor":
        model = SGDRegressor()

        # Initially define the parameters for SGD Regressor
        parameters = {
            "loss": ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            "penalty": ["l2", "l1", "elasticnet"],
            "alpha": tol,
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "shuffle": [True, False]
        }

        # Conditionally modify the 'l1_ratio', 'eta0', and 'max_iter' parameters
        if 'elasticnet' in parameters['penalty']:
            l1_ratio = np.arange(0.1, 1, 0.1)
        else:
            l1_ratio = None
        parameters["l1_ratio"] = l1_ratio

        if any(rate in parameters['learning_rate'] for rate in ['constant', 'invscaling', 'adaptive']):
            eta0 = [0.01, 0.1, 0.5]
        else:
            eta0 = None
        parameters["eta0"] = eta0

        if 'invscaling' in parameters['learning_rate']:
            max_iter_value = max_iter
        else:
            max_iter_value = None
        parameters["max_iter"] = max_iter_value

    elif algorithm == "Huber Regressor":
        model = HuberRegressor()

        # Define the parameters for Huber Regressor
        parameters = {
            "epsilon": [1.35, 1.5, 1.75],
            "max_iter": max_iter,
            "alpha": tol,
            "fit_intercept": [True, False],
        }

    elif algorithm == "Lars":
        model = Lars()

        # Define the parameters for Lars
        parameters = {
            "n_nonzero_coefs": [1, 5, 10],
            "eps": [1e-4, 1e-3, 1e-2],
        }

    elif algorithm == "RANSAC Regressor":
        model = RANSACRegressor()

        # Define the parameters for RANSAC Regressor
        parameters = {
            "min_samples": [None, 0.1, 0.5, 1.0],
            "max_trials": [100, 500, 1000],
            "residual_threshold": [1.0, 2.0, 3.0],
        }

    elif algorithm == "Theil-Sen Regressor":
        model = TheilSenRegressor()

        # Define the parameters for Theil-Sen Regressor
        parameters = {
            "max_subpopulation": [100, 500, 1000],
            "max_iter": max_iter,
            "tol": tol,
        }

    elif algorithm == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor()
        parameters = {
            "loss": ["huber", "quantile"],
            "learning_rate": learning_rate,
            "n_estimators": np.arange(10, 310, 50),
            "subsample": [1.0, 0.9, 0.8],
            "criterion": ["friedman_mse", "squared_error", "absolute error"],
            "max_depth": [3, 5, 7],
        }

    elif algorithm == "Logistic Regression":
        model = LogisticRegression()

        # Define a list of all valid solver-penalty combinations
        solver_penalty_combinations = [
            {'solver': 'lbfgs', 'penalty': 'l2'},
            {'solver': 'lbfgs', 'penalty': 'none'},
            {'solver': 'liblinear', 'penalty': 'l1'},
            {'solver': 'liblinear', 'penalty': 'l2'},
            {'solver': 'newton-cg', 'penalty': 'l2'},
            {'solver': 'newton-cg', 'penalty': 'none'},
            {'solver': 'sag', 'penalty': 'l2'},
            {'solver': 'sag', 'penalty': 'none'},
            {'solver': 'saga', 'penalty': 'l1'},
            {'solver': 'saga', 'penalty': 'l2'},
            {'solver': 'saga', 'penalty': 'elasticnet'},
            {'solver': 'saga', 'penalty': 'none'}
        ]

        # Expand the parameter grid
        parameters = []
        for combo in solver_penalty_combinations:
            params = {
                "C": np.logspace(-2, 1, 4),
                "max_iter": max_iter,
                "tol": tol
            }
            params.update(combo)
            parameters.append(params)

    elif algorithm == "MLPClassifier":
        model = MLPClassifier()
        parameters = {
            "activation": ['relu', 'identity', 'logistic', 'tanh'],
            "hidden_layer_sizes": [(10,), (20,), (10, 10)],
            "alpha": learning_rate,
            "solver": ["sgd", "adam"],
            "learning_rate_init": learning_rate,
            "momentum": [0.9, 0.95, 0.99],
            "max_iter": max_iter
        }
    elif algorithm == "BernoulliRBM":
        model = BernoulliRBM()

        # Define the parameters for BernoulliRBM
        parameters = {
            # Number of components or hidden units
            "n_components": [10, 20, 30],
            "learning_rate": [0.1, 0.01, 0.001],
            "n_iter": [10, 20, 30],  # Number of iterations
            "batch_size": [16, 32, 64],  # Mini-batch size
        }

    elif algorithm == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
        parameters = {
            "min_samples_leaf": [1, 2, 3, 4, 10],
            "criterion": ["gini", "entropy"],
            "max_features": ['sqrt', 'log2'],
            "min_impurity_decrease": [0.1, 0.2],
            "class_weight": ['balanced'],
            "ccp_alpha": [0.1, 1, 0],
        }
    elif algorithm == "Random Forest Classifier":
        model = RandomForestClassifier()
        parameters = {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 3],
            "min_samples_leaf": [3, 4],
            "criterion": ["gini", "entropy"],
            "bootstrap": [True],
            "max_features": ['sqrt', 'log2', None]
        }
    elif algorithm == "SVC":
        model = SVC()
        parameters = {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["auto", "scale"],
            "tol": tol,
        }

    elif algorithm == "Isolation Forest":
        model = IsolationForest()
        parameters = {
            "n_estimators": n_estimators,
            "contamination": [0.1, 0.2, 0.3],
            "max_features": [5, 10, 15, 20],
            "n_jobs": [-1]
        }
    elif algorithm == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier()
        parameters = {
            "loss": ["log_loss"],
            "learning_rate": [0.1, 0.01, 0.001],
            "n_estimators": n_estimators,
            "subsample": [1, 0.9, 0.8],  # Example additional parameter
            "criterion": ["friedman_mse", "squared_error"],
            "max_depth": [3, 4, 6, 8, 10],
            "min_samples_split": [2, 5, 10],  # Example additional parameter
            # Corrected max_features for feature sampling
            "max_features": ['sqrt', 'log2', None],
            "min_weight_fraction_leaf": [0, 0.1, 0.2],
            "max_leaf_nodes": [None, 10, 20, 30],
            "min_impurity_decrease": [0, 0.01, 0.1],
            "validation_fraction": [0.1, 0.2],
            "n_iter_no_change": [None, 10, 20],
            "tol": [0.0001, 0.001],
            "ccp_alpha": [0, 0.01, 0.1]
        }

    elif algorithm == "Extra Tree Classifier":
        model = ExtraTreeClassifier()
        parameters = {
            "max_depth": [3, 4, 6, 8, 10],
            "min_samples_split": [3, 4, 6],
            "criterion": ["gini", "entropy"],
            # Corrected max_features for feature sampling
            "max_features": ['sqrt', 'log2', None],
            "splitter": ["best", "random"],
            "min_impurity_decrease": [0.0, 0.1],  # Example values
        }

    elif algorithm == "XGBoost Classifier":
        model = XGBClassifier()
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
            "n_jobs": [-1]
        }
    elif algorithm == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
        parameters = {
            "n_neighbors": [3, 5, 11, 19],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [30, 40, 50],
            "p": [1, 2],
            "n_jobs": [-1]
        }
    elif algorithm == "AdaBoost Classifier":
        model = AdaBoostClassifier()
        parameters = {
            "n_estimators": np.arange(10, 310, 50),
            "learning_rate": learning_rate,
            "algorithm": ["SAMME", "SAMME.R"]
        }

    elif algorithm == "Gaussian Process Classifier":
        from sklearn.gaussian_process.kernels import RBF, Matern
        model = GaussianProcessClassifier()
        Rbf_ = 1.0 * RBF(length_scale=1.0)
        Mattern_ = 1.0 * Matern(length_scale=1.0)
        parameters = {
            "kernel": [Rbf_, Mattern_],
            "n_restarts_optimizer": [0, 1, 2],
            "max_iter_predict": np.arange(100, 500, 100),
            "optimizer": ["fmin_l_bfgs_b"],
            "n_jobs": [-1],
            "copy_X_train": [True, False],
            "multi_class": ["one_vs_rest", "one_vs_one"],
            # None for no seed, 42 for a specific seed
            "random_state": [0, 10, 20, 50]
        }
    elif algorithm == "Quadratic Discriminant Analysis":
        model = QuadraticDiscriminantAnalysis()
        parameters = {
            "reg_param": np.arange(0.0, 0.10, 0.1),
            "store_covariance": [True, False],
            "tol": np.logspace(-3, -1, 3),
        }
    elif algorithm == "SGD Classifier":
        model = SGDClassifier()

        # Initially define the parameters for SGD Classifier
        parameters = {
            "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            "penalty": ["l2", "l1", "elasticnet"],
            "alpha": tol,
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "shuffle": [True, False],
            "n_jobs": [-1]
        }

        # Conditionally modify the 'l1_ratio', 'eta0', and 'max_iter' parameters
        if 'elasticnet' in parameters['penalty']:
            l1_ratio_options = np.arange(0.1, 1, 0.1)
        else:
            l1_ratio_options = None
        parameters["l1_ratio"] = l1_ratio_options

        if any(rate in parameters['learning_rate'] for rate in ['constant', 'invscaling', 'adaptive']):
            eta0_options = [0.01, 0.1, 0.5]
        else:
            eta0_options = None
        parameters["eta0"] = eta0_options

        # Assuming 'max_iter' is predefined and applicable to all learning_rate options
        parameters["max_iter"] = max_iter

    else:
        raise ValueError("Invalid algorithm: {}".format(algorithm))

    # Define the Pipeline only if the algorithm is one of the specified regressions
    if algorithm in ["Linear Regression"]:
        try:
            # Wrap the existing model in a pipeline that includes feature selection
            model = Pipeline([
                ('feature_selection', SelectFromModel(LassoCV())),
                ('regression', model)
            ])
            # Add or update parameters for the feature_selection step
            # Ensure existing parameters are updated to work with the pipeline
            new_parameters = {
                f'regression__{key}': value for key, value in parameters.items()}
            new_parameters['feature_selection__estimator__cv'] = [
                5]  # Example CV parameter for LassoCV
            new_parameters['feature_selection__threshold'] = [
                1e-5, 1e-4, 1e-3]  # Feature selection thresholds
            parameters = new_parameters
        except NotFittedError:
            # Handle cases where the model provided is not fitted or is incompatible
            print("Model not fitted or incompatible with SelectFromModel.")

    return model, parameters
