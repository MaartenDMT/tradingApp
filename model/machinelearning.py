import threading

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import (GradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import column_or_1d
from xgboost import XGBClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from skopt import BayesSearchCV

from util.utils import array_min2d


class MachineLearning:
    def __init__(self, exchange, symbol, logger) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.logger = logger
    
    def predict(self, model, t='1m', symbol='BTC/USDT') -> int:
        # Fetch the current data for the symbol
        data = self.exchange.fetch_ohlcv(symbol, timeframe=t, limit=1)
        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        data = np.array(df)
        # Calculate the technical indicators
        df['rsi'] = ta.rsi(df.close)
        df['macd'] = ta.macd(df.close)
        df['moving_average'] = ta.sma(df.close, length=50)
        # Create the features array
        features = np.column_stack((df.open.fillna(0), df.high.fillna(0), df.low.fillna(0), df.close.fillna(0), df.volume.fillna(0)))
        # Make a prediction using the model
        prediction = model.predict(features[-1].reshape(1, -1))[0]
        return prediction
  

    def evaluate_model(self, name, model, X_test, y_test):
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        if name in ['Linear Regression', 'SVR']:
            accuracy = r2_score(y_test, y_pred)
        else:
            # Calculate the accuracy of the model
            accuracy = accuracy_score(y_test, y_pred)
        
        # Print the accuracy of the model
        print(f"Accuracy: {accuracy} of model: {model}")
        
        # Return the accuracy of the model
        return accuracy

    def save_model(self, model, accuracy):
        # Save the model if it has high accuracy
        path = r'data/ml/'
        if accuracy > 0.7:
            joblib.dump(model.best_estimator_, f'{path}trained_{model.best_estimator_.__class__.__name__}-{accuracy}.pkl')
            
    def load_model(self, filename):
        path = r'data/ml/'
        file = f'{path}{filename}'
        try:
            # Open the file and load the model
            model = joblib.load(file)
        except FileNotFoundError:
            # name = 'Decision Tree Classifier'
            # model = self.train_evaluate_and_save_model(name)
            self.logger.info(f'there is no file called: {file}')

        return model



    def train_evaluate_and_save_model(self, model:str):
        # Fetch the historical data for the symbol
        data = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=500)
        df = pd.DataFrame(data, columns=['date','open', 'high', 'low', 'close', 'volume'])
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        # Calculate the technical indicators
        rsi = ta.rsi(df.close)
        macd = ta.macd(df.close)
        moving_average = ta.sma(df.close, length=50)
        # Create the features array
        features = np.column_stack((rsi.fillna(0), macd.fillna(0), moving_average.fillna(0), df.open.fillna(0), df.high.fillna(0), df.low.fillna(0), df.volume.fillna(0)))
        
        if model in ["SVC", "Random Forest Classifier", 'Decision Tree Classifier', 
                     'Extra Tree Classifier', 'Logistic Regression', 'MLPClassifier', 
                     'Gradient Boosting Classifier']:
            
            # Divide labels into three categories: low, medium, and high
            labels = ['low', 'medium','high']
            df['label_category'] = pd.qcut(df['close'].astype(float), q=5, labels=labels, duplicates='drop')
            print(df['label_category'])
            # Create labels using the label categories
            labels = df['label_category']
        else:
            labels = df['close']

        # Initialize the scaler
        scaler = StandardScaler()
        scaler.fit(features)
        features_scaled = scaler.transform(features)
        
        label_scaler = LabelEncoder()
        label_scaler.fit(labels)

        labels_scaled = label_scaler.transform(labels)
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_scaled, test_size=0.2)
        

        if model in ['Random Forest Classifier','Gradient Boosting Classifier','SVC', 
                     'Logistic Regression','Decision Tree Classifier','MLPClassifier',
                     "SVR", "Extra Tree Classifier", 'XGBoost Classifier', 
                     'Linear Regression', "Radius Neighbors Classifier"]:
            
            y_train = column_or_1d(y_train, warn=True)
            y_test = column_or_1d(y_test, warn=True)
                
        else:
            y_train = array_min2d(y_train)
            y_test = array_min2d(y_test)
            
        # Train the model
        trainer = MLModelTrainer(model, self.logger)
        trained_model = trainer.train(X_train, y_train)
            
        # Evaluate the model
        accuracy = self.evaluate_model(model , trained_model, X_test, y_test)

        
        #save the model
        self.save_model(trained_model,accuracy)
        self.logger.info(f"the {model} model has been trained evaluated {accuracy} and saved selected")
        
        
        return trained_model
      
    def selected_labels_features_train(self, model:str, X, y):
        trainer = MLModelTrainer(model, self.logger)
        trained_model = trainer.train(X, y)
        
        return trained_model
    
    def train_evaluate_and_save_model_thread(self, model:str) -> None:
        # Create a new thread
        t = threading.Thread(target=self.train_evaluate_and_save_model, args=(model,))
        t.setDaemon(True)
        # Start the thread
        t.start()
        

    def selected_labels_features_train_thread(self, model:str, X, y)-> None:
        # Create a new thread
        t = threading.Thread(target=self.selected_labels_features_train, args=(model,X,y,))
        t.setDaemon(True)
        # Start the thread
        t.start()



class MLModelTrainer:
    def __init__(self, algorithm, logger) -> None:
        self.algorithm = algorithm
        self.logger = logger

    def train(self, X, y):
        if self.algorithm == "Linear Regression":
            model = LinearRegression()
            parameters = {
                "fit_intercept": [True, False],
                "copy_X": [True, False],
                "n_jobs": [-1, 1]
            }
        elif self.algorithm == "Logistic Regression":
            model = LogisticRegression()
            parameters = {
                "penalty": ["l2"],
                "C": [0.1, 1.0, 10.0],
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                "max_iter": [300, 400, 500, 600]
            }
        elif self.algorithm == "MLPClassifier":
            model = MLPClassifier()
            parameters = {
                "hidden_layer_sizes": [(10,), (20,), (10, 10)],
                "alpha": [0.1, 1.0, 10.0],
                "solver": ["sgd", "adam", "lbfgs"],
                "learning_rate_init": [0.001, 0.01, 0.1],
                "momentum": [0.9, 0.95, 0.99],
            }
        elif self.algorithm == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
            parameters = {
                "max_depth": [3, 5, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 3, 4, 5],
                "criterion": ["gini", "entropy"]
            }
        elif self.algorithm == "Random Forest Classifier":
            model = RandomForestClassifier()
            parameters = {
                "n_estimators": [10, 50, 100, 200, 300],
                "max_depth": [2, 4, 6, 8, 10],
                "min_samples_split": [2, 4, 6, 8, 10],
                "min_samples_leaf": [1, 2, 3, 4, 5],
                "criterion": ["gini", "entropy"]
            }
        elif self.algorithm == "SVC":
                model = SVC()
                parameters = {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "degree": [2, 3, 4, 5],
                    "gamma": ["auto", "scale"]
            }
        elif self.algorithm == "SVR":
            model = SVR()
            parameters = {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "degree": [2, 3, 4, 5],
                "gamma": ["auto", "scale"]
            }
        elif self.algorithm == "Isolation Forest":
            model = IsolationForest()
            parameters = {
                "n_estimators": [10, 50, 100, 200, 300],
                "max_samples": ["auto", "None"],
                "contamination": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        elif self.algorithm == "Gradient Boosting Classifier":
            model = GradientBoostingClassifier()
            parameters = {
                "loss": ["deviance", "exponential"],
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [10, 50, 100, 200, 300],
                "max_depth": [2, 4, 6, 8, 10]
            }
        elif self.algorithm == "Extra Tree Classifier":
            model = ExtraTreeClassifier ()
            parameters = {
                "max_depth": [2, 4, 6, 8, 10],
                "min_samples_split": [2, 4, 6, 8, 10],
                "min_samples_leaf": [1, 2, 3, 4, 5],
                "criterion": ["gini", "entropy"]
            }
        elif self.algorithm == "XGBoost Classifier":
            model = XGBClassifier()
            parameters = {
                "max_depth": [2, 4, 6, 8, 10],
                "eta": [0.1,0.2,0.3,0.4,0.5],
                "objective": ["binary:logistic","multi:softmax"],
                "num_class": [1,2,3,4]
            }
        elif self.algorithm == "Gaussian Naive Bayes":
            model = GaussianNB()
            parameters = {
                'priors': [None, [0.1, 0.9], [0.2, 0.8]],
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        elif self.algorithm == "Radius Neighbors Classifier":
            model = RadiusNeighborsClassifier()
            parameters = {
                'radius': [1, 2, 3, 4, 5,6,7,8,9,10],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [30, 40, 50],
                'p': [1, 2],
                'outlier_label': ['outliners']
            }
        else:
            raise ValueError("Invalid algorithm: {}".format(self.algorithm))
            
        best_estimator = self.parallelize_search(model, parameters, X, y)
            
        best_params = best_estimator.best_params_
        best_f1_score = best_estimator.best_score_

        # print the best parameters and best score
        self.logger.info(f"the custom: parallelize search best params are {best_params}")
        self.logger.info(f"the custom: parallelize search best score is {best_f1_score}")
        self.logger.info(f"the custom: parallelize search best estimator is {best_estimator}")

        # return the best model
        return best_estimator

    def parallelize_search(self, estimator, param_grid, X, y):
        grid_search_estimator = GridSearchCV(estimator, param_grid, cv=5)
        random_search_estimator = RandomizedSearchCV(estimator, param_grid, cv=5)
        bayes_search_estimator = BayesSearchCV(estimator, param_grid, cv=5)
        
        with ThreadPoolExecutor() as executor:
            grid_search = executor.submit(grid_search_estimator.fit, X, y)
            random_search = executor.submit(random_search_estimator.fit, X, y)
            bayes_search = executor.submit(bayes_search_estimator.fit, X, y)
        
            results = [result.result() for result in as_completed([grid_search, random_search, bayes_search])]
       
        best_estimator = max(results, key=lambda x: x.best_score_ if isinstance(x, (GridSearchCV, RandomizedSearchCV, BayesSearchCV)) else x.best_score)
        
        print(best_estimator)
        
        return best_estimator