import threading

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import (GradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import column_or_1d

from util.utils import array_min2d


class MachineLearning:
    def __init__(self, exchange, symbol, logger):
        self.exchange = exchange
        self.symbol = symbol
        self.logger = logger
    
    def predict(self, model):
        # Fetch the current data for the symbol
        data = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=1)
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
  

    def evaluate_model(self, model, X_test, y_test):
            # Make predictions on the test data
            y_pred = model.predict(X_test)
            
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
            joblib.dump(model, f'{path}trained_model{model}-{accuracy}.pkl')
            
    def load_model(self, filename):
        try:
            # Open the file and load the model
            model = joblib.load(filename)
        except FileNotFoundError:
            name = 'Decision Tree Classifier'
            model = self.train_evaluate_and_save_model(name)

            
        return model



    def train_evaluate_and_save_model(self, model:str):
        # Fetch the historical data for the symbol
        data = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=500)
        df = pd.DataFrame(data, columns=['date','open', 'high', 'low', 'close', 'volume'])
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        data = np.array(df)
        # Calculate the technical indicators
        rsi = ta.rsi(df.close)
        macd = ta.macd(df.close)
        moving_average = ta.sma(df.close, length=50)
        # Create the features array
        features = np.column_stack((rsi.fillna(0), macd.fillna(0), moving_average.fillna(0)))
        
        if model in ["SVC", "Random Forest Classifier", 'Decision Tree Classifier', 
                     'Gradient Boosting Classifier', 'Extra Tree Classifier', 
                     'Logistic Regression', 'MLPClassifier']:
            # Divide labels into three categories: low, medium, and high
            bins = [float(df['close'].min() - 1), float(df['close'].quantile(q=0.33)), float(df['close'].quantile(q=0.66)), float(df['close'].max() + 1)]
            labels = ['low', 'medium', 'high']
            df['label_category'] = pd.cut(df['close'].astype(float), bins=bins, labels=labels)
            # Create labels using the label categories
            labels = df['label_category']
        else:
            labels = df['close']

        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
        

        
        # Initialize the scaler
        scaler = StandardScaler()
        # Fit the scaler to the training data
        if model in ['SVC',"SVR", "Random Forest Classifier", 'Decision Tree Classifier', 
                     'Gradient Boosting Classifier', 'Extra Tree Classifier', 
                     'Logistic Regression', 'Linear Regression','MLPClassifier']: 
            label_scaler_train = LabelEncoder()
            label_scaler_test = LabelEncoder()
            scaler.fit(X_train)
            if model in ['Random Forest Classifier','Gradient Boosting Classifier','SVC', 'Logistic Regression','Decision Tree Classifier','MLPClassifier',"SVR", "Extra Tree Classifier"]:
                y_train = column_or_1d(y_train, warn=True)
                y_test = column_or_1d(y_test, warn=True)
                
            else:
                y_train = array_min2d(y_train)
                y_test = array_min2d(y_test)

            label_scaler_train.fit(y_train)
            label_scaler_test.fit(y_test)
            # Transform the training and test data
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            y_train_scaled = label_scaler_train.transform(y_train)
            y_test_scaled = label_scaler_test.transform(y_test)

            
            # Train the model
            trainer = MLModelTrainer(model, self.logger)
            trained_model = trainer.train(X_train_scaled, y_train_scaled)
            
            # Evaluate the model
            accuracy = self.evaluate_model(trained_model, X_test_scaled, y_test_scaled)
        else: 
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            # Train the model
            trainer = MLModelTrainer(model, self.logger)
            trained_model = trainer.train(X_train_scaled, y_train)
            
            # Evaluate the model
            accuracy = self.evaluate_model(trained_model, X_test, y_test)
        
        #save the model
        self.save_model(trained_model,accuracy)
        self.logger.info(f"the {model} model has been trained evaluated {accuracy} and saved selected")
        
        
        return trained_model
      
    def selected_labels_features_train(self, model:str, X, y):
        trainer = MLModelTrainer(model, self.logger)
        trained_model = trainer.train(X, y)
        
        return trained_model
    
    def train_evaluate_and_save_model_thread(self, model:str):
        # Create a new thread
        t = threading.Thread(target=self.train_evaluate_and_save_model, args=(model,))
        t.setDaemon(True)
        # Start the thread
        t.start()
        

    def selected_labels_features_train_thread(self, model:str, X, y):
        # Create a new thread
        t = threading.Thread(target=self.selected_labels_features_train, args=(model,X,y,))
        t.setDaemon(True)
        # Start the thread
        t.start()



class MLModelTrainer:
    def __init__(self, algorithm, logger):
        self.algorithm = algorithm
        self.logger = logger
        print(algorithm)
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
                "penalty": ["none", "l2"],
                "C": [0.1, 1.0, 10.0],
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                "max_iter": [50, 100, 200, 300]
            }
        elif self.algorithm == "MLPClassifier":
            model = MLPClassifier()
            parameters = {
                "hidden_layer_sizes": [(10,), (20,), (10, 10)],
                "alpha": [0.1, 1.0, 10.0]
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

        else:
            raise ValueError("Invalid algorithm: {}".format(self.algorithm))

        # check if the model has a score method

        if self.algorithm in ['Logistic Regression' , 'MLPClassifier' , 'Decision Tree Classifier'  ,'Random Forest Classifier','Extra Tree Classifier']:
            # use grid search with the model's score method
            grid_search = GridSearchCV(model, parameters, cv=5)
        else:
            # define a custom scoring function
            def custom_scoring(estimator, X, y):
                # calculate some metric using the estimator and the data
                y_pred = estimator.predict(X)
                accuracy = (y_pred == y).mean()
                # precision = precision_score(y, y_pred)
                # recall = recall_score(y, y_pred)
                # f1 = f1_score(y, y_pred)
                return accuracy


            custom_scorer = make_scorer(custom_scoring,greater_is_better=True)
            # use grid search with the custom scoring function
            grid_search = GridSearchCV(model, parameters, cv=5, scoring=lambda estimator, X, y: custom_scoring(estimator, X, y))

        # fit the grid search object to the training data
        grid_search.fit(X, y)

        # print the best parameters and best score
        self.logger.info(f"the grid search best params are {grid_search.best_params_}")
        self.logger.info(f"the grid search best score is {grid_search.best_score_}")

        # return the best model
        return grid_search.best_estimator_

# trainer = MLModelTrainer(selected_algorithm)
# model = trainer.train(X, y)