import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.metrics import accuracy_score, make_scorer, r2_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import column_or_1d
from skopt import BayesSearchCV

from util.ml_models import get_model
from util.utils import array_min2d


class MachineLearning:
    def __init__(self, exchange, symbol, logger) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.logger = logger

    def predict(self, model, t='30m', symbol='BTC/USDT') -> int:
        # TODO: ta.macd problem with the return wants to give 3 but you need only one

        # Fetch the current data for the symbol
        data = self.exchange.fetch_ohlcv(symbol, timeframe=t, limit=100)
        df = pd.DataFrame(
            data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        data = np.array(df)

        # Calculate the technical indicators
        df['rsi'] = ta.rsi(df.close)
        df['moving_average'] = ta.sma(df.close, length=50)

        # Create the features array
        features = np.column_stack((df.rsi.fillna(0), df.moving_average.fillna(
            0), df.open.fillna(0), df.high.fillna(0), df.low.fillna(0), df.volume.fillna(0)))

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
            joblib.dump(model.best_estimator_,
                        f'{path}trained_{model.best_estimator_.__class__.__name__}-{accuracy}.pkl')

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

    def process_features(self, df):
        # Calculate RSI
        rsi = ta.rsi(df['close'])

        # Calculate moving average
        moving_average = ta.sma(df['close'], length=50)

        # Combine all the selected features into a single DataFrame
        processed_features = pd.concat(
            [rsi, moving_average, df[['open', 'high', 'low', 'volume']]], axis=1)

        # Fill missing values in the entire DataFrame
        processed_features = processed_features.fillna(0)

        # Return the processed features as a NumPy array
        return processed_features.to_numpy()

    def process_labels(self, df, model, n_candles=2):
        if model in ["SVC", "Random Forest Classifier", "Extra Tree Classifier",
                     "Logistic Regression", "MLPClassifier", "Gradient Boosting Classifier"]:
            # Shift closing prices by -2 to compare with the price two candles ahead
            df['future_close'] = df['close'].shift(-n_candles)

            # Define labels: 1 for long, 0 for do nothing, -1 for short
            df['label'] = np.sign(df['future_close'] - df['close'])

            self.logger.info(df['label'])
            return df['label']
        else:
            return df['close']

    def train_evaluate_and_save_model(self, model: str):
        # Fetch and preprocess data
        data = self.exchange.fetch_ohlcv(
            self.symbol, timeframe='30m', limit=5_000)
        df = pd.DataFrame(
            data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        features = self.process_features(df)
        labels = self.process_labels(df, model, n_candles=5)

        # Drop the last two rows as they won't have valid labels
        features = features[:-2]
        labels = labels[:-2]

        # Scale features and labels
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        label_scaler = LabelEncoder()
        labels_scaled = label_scaler.fit_transform(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels_scaled, test_size=0.2)

        models_needing_1d = ["Random Forest Classifier", "Gradient Boosting Classifier", "SVC",
                             "Logistic Regression", "Decision Tree Classifier", "MLPClassifier",
                             "SVR", "Extra Tree Classifier", "XGBoost Classifier",
                             "Linear Regression", "Radius Neighbors Classifier"]

        if model in models_needing_1d:
            y_train = column_or_1d(y_train, warn=True)
            y_test = column_or_1d(y_test, warn=True)
        else:
            y_train = array_min2d(y_train)
            y_test = array_min2d(y_test)

        # # MLPCLASSEFIER
        # if model == "MLPClassifier":
        #     X_train = np.asarray(X_train, dtype=object)

        # Train, evaluate, and save the model
        trainer = MLModelTrainer(model, self.logger)
        trained_model = trainer.train(X_train, y_train)

        accuracy = self.evaluate_model(model, trained_model, X_test, y_test)

        self.save_model(trained_model, accuracy)
        self.logger.info(
            f"The {model} model has been trained, evaluated {accuracy}, and saved.")

        return trained_model

    def selected_labels_features_train(self, model: str, X, y):
        trainer = MLModelTrainer(model, self.logger)
        trained_model = trainer.train(X, y)

        return trained_model

    def train_evaluate_and_save_model_thread(self, model: str) -> None:
        # Create a new thread
        t = threading.Thread(
            target=self.train_evaluate_and_save_model, args=(model,))
        t.setDaemon(True)
        # Start the thread
        t.start()

    def selected_labels_features_train_thread(self, model: str, X, y) -> None:
        # Create a new thread
        t = threading.Thread(
            target=self.selected_labels_features_train, args=(model, X, y,))
        t.setDaemon(True)
        # Start the thread
        t.start()


class MLModelTrainer:
    def __init__(self, algorithm, logger) -> None:
        self.algorithm = algorithm
        self.logger = logger

    def train(self, X, y):

        model, parameters = get_model(self.algorithm)

        # Create a scorer from the custom scoring function
        # Create a scorer from the custom scoring function using lambda to handle n_candles
        monetary_scorer = make_scorer(lambda y_true, y_pred: self.future_score(
            y_true, y_pred, n_candles=5), greater_is_better=True)

        best_estimator = self.parallelize_search(
            model, parameters, X, y, monetary_scorer)

        best_params = best_estimator.best_params_
        best_f1_score = best_estimator.best_score_

        # print the best parameters and best score
        self.logger.info(
            f"the custom: parallelize search best params are {best_params}")
        self.logger.info(
            f"the custom: parallelize search best score is {best_f1_score}")
        self.logger.info(
            f"the custom: parallelize search best estimator is {best_estimator}")

        # return the best model
        return best_estimator

    def parallelize_search(self, estimator, param_grid, X, y, scorer):
        grid_search_estimator = GridSearchCV(
            estimator, param_grid, cv=5, scoring=scorer)
        random_search_estimator = RandomizedSearchCV(
            estimator, param_grid, cv=5, scoring=scorer)
        bayes_search_estimator = BayesSearchCV(
            estimator, param_grid, cv=5, scoring=scorer)

        with ThreadPoolExecutor() as executor:
            grid_search = executor.submit(
                grid_search_estimator.fit, X, y)
            random_search = executor.submit(
                random_search_estimator.fit, X, y)
            bayes_search = executor.submit(
                bayes_search_estimator.fit, X, y)

            results = [result.result() for result in as_completed(
                [grid_search, random_search, bayes_search])]

        best_estimator = max(results, key=lambda x: x.best_score_ if isinstance(
            x, (GridSearchCV, RandomizedSearchCV, BayesSearchCV)) else x.best_score)

        self.logger.info(best_estimator)

        return best_estimator

    @staticmethod
    def future_score(y_true, y_pred, n_candles=2):
        total_return = 0
        position = 0  # -1 for short, 0 for no position, 1 for long
        entry_price = 0

        # Check the lengths of y_true and y_pred and adjust if necessary
        length_to_iterate = min(len(y_pred), len(y_true) - n_candles)
        if length_to_iterate < len(y_pred):
            # Log a warning or adjust as necessary
            ...

        # Iterate only up to the length where we have enough future data
        for i in range(length_to_iterate):
            if position != 0:  # if holding a position, long or short
                total_return += (y_true[i + n_candles] -
                                 entry_price) * position
                position = 0  # close the position

            if y_pred[i] != 0:  # 1 for long, -1 for short
                entry_price = y_true[i]
                position = y_pred[i]

        # Check if we are holding any position at the end
        if position != 0:
            total_return += (y_true[-1] - entry_price) * position

        return total_return

    @staticmethod
    def spot_score(y_true, y_pred, n_candles=2):
        total_return = 0
        holding_stock = False
        buy_price = 0

        # Check the lengths of y_true and y_pred and adjust if necessary
        length_to_iterate = min(len(y_pred), len(y_true) - n_candles)
        if length_to_iterate < len(y_pred):
            # Log a warning or adjust as necessary
            ...

        # Iterate only up to the length where we have enough future data
        for i in range(length_to_iterate):
            if holding_stock:
                # Sell the stock after two candles and calculate the return
                total_return += y_true[i + n_candles] - buy_price
                holding_stock = False

            if y_pred[i] == 1:  # Assuming 1 is a signal to buy, and 0 is a signal to hold/sell
                buy_price = y_true[i]
                holding_stock = True

        # Check if we are holding any stock at the end
        if holding_stock:
            total_return += y_true[-1] - buy_price

        return total_return
