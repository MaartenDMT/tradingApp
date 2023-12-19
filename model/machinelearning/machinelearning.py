import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pandas_ta as ta
import seaborn as sns
from joblib import dump, load
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, explained_variance_score,
                             make_scorer, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

import util.loggers as loggers
from util.ml_models import get_model
from util.ml_util import (classifier, column_1d, future_score_clas,
                          future_score_reg, regression, spot_score_clas,
                          spot_score_reg)
from util.utils import array_min2d, tradex_features

logger = loggers.setup_loggers()


class MachineLearning:
    def __init__(self, exchange, symbol) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.logger = logger['model']

    def predict(self, model, df=None, t='30m', symbol='BTC/USDT') -> int:
        # TODO: predict(self, model, features)
        # the features needs to be a from my trade-x thingy
        try:
            # Fetch the current data for the symbol
            data = self.exchange.fetch_ohlcv(symbol, timeframe=t, limit=500)
            df = pd.DataFrame(
                data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

            # Use the process_features method to get the features for prediction
            features = self.process_features(df)

            # If a scaler was used during training, load the pre-fitted scaler and transform the features
            # Make sure the scaler is saved after fitting on training data
            # load the scaler fitted on training data
            scaler = self.load_pretrained_scaler()
            features = scaler.transform(features)

            # Make a prediction using the model
            prediction = model.predict(features[-1].reshape(1, -1))[0]

            # Logging the prediction result
            self.logger.info(
                f"Prediction for {symbol} at timeframe {t}: {prediction}")

            return prediction
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            return 0  # return a default value or handle according to your use case

    def evaluate_model(self, name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        plot_dir = 'data/plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Create a new figure and a subplot axis
        fig, ax = plt.subplots(figsize=(10, 6))

        accuracy = None
        if name in regression:
            # Calculate additional regression metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            evs = explained_variance_score(y_test, y_pred)

            # Print the metrics
            self.logger.info(f"R2 Score: {r2}")
            self.logger.info(f"Mean Absolute Error: {mae}")
            self.logger.info(f"Mean Squared Error: {mse}")
            self.logger.info(f"Root Mean Squared Error: {rmse}")
            self.logger.info(f"Explained Variance Score: {evs}")

           # Visualize the results
            ax.scatter(y_test, y_pred)
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.set_title('True Values vs Predictions', loc='right')

            # Visualize the residuals in a new subplot
            ax_res = ax.twinx()  # Create a twin Y axis sharing the same X axis
            residuals = y_test - y_pred
            ax_res.hist(residuals, alpha=0.5, color='orange')
            ax_res.set_ylabel('Frequency')
            ax_res.set_title('Residual Histogram', loc='left')

            fig.suptitle('Regression Analysis')

            # For regression models, you might want to return R2 score or another metric as "accuracy"
            accuracy = r2

        else:

            # Calculate additional classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred) if len(
                set(y_test)) == 2 else "Not Applicable"
            self.logger.info(f"Accuracy: {accuracy}")
            self.logger.info(f"AUC-ROC: {auc}")
            self.logger.info(
                f"Classification Report:\n{classification_report(y_test, y_pred)}")

            # confusion matrix
            conf_mat = confusion_matrix(y_test, y_pred)

            # Visualize the results
            sns.heatmap(conf_mat, annot=True, fmt='d', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix', loc='right')

            if len(set(y_test)) == 2:
                # ROC Curve for binary classification in a new subplot
                ax_roc = ax.twinx()  # Create a twin Y axis sharing the same X axis
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                ax_roc.plot(fpr, tpr, color='orange')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('ROC Curve', loc='left')

            fig.suptitle('Classification Analysis')

        if name in regression:
            # Save the plot
            plt.savefig(f'{plot_dir}/{name}_plot.png')
        else:
            # Save the plot
            plt.savefig(f'{plot_dir}/{name}_confusion.png')

        # Show or close the plot
        if threading.current_thread() == threading.main_thread():
            plt.show()
        else:
            plt.close()

        return accuracy

    def save_model(self, model, accuracy, score=None):
        # Save the model if it has high accuracy
        path = r'data/ml/2020/'
        params_path = r'data/csv/params_accuracy.csv'

        if accuracy > 0.62:
            joblib.dump(model.best_estimator_,
                        f'{path}trained_{model.best_estimator_.__class__.__name__}{accuracy*100}.p')

        # Save the parameters and accuracy
        model_params_accuracy = {
            "Model": model.best_estimator_.__class__.__name__,
            "Params": str(model.best_params_),
            "Accuracy": accuracy,
            "Money": score
        }

        # Check if the CSV file already exists
        if os.path.exists(params_path):
            # If it exists, load the existing data
            df = pd.read_csv(params_path)
            df = df.append(model_params_accuracy, ignore_index=True)
        else:
            # If it doesn't exist, create a new DataFrame
            df = pd.DataFrame([model_params_accuracy])

        # Save the updated DataFrame to the CSV file
        df.to_csv(params_path, index=False)

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

        processed_features = tradex_features(self.symbol, df)

        self.column_names = processed_features.columns.tolist()

        # Fill missing values in the entire DataFrame
        processed_features = processed_features.fillna(0)

        # Return the processed features as a NumPy array
        return processed_features.to_numpy()

    def process_labels(self, df, model, n_candles=2, threshold=0.08):
        if model in classifier:
            # Shift closing prices by -2 to compare with the price two candles ahead
            df['future_close'] = df['close'].shift(-n_candles)

            # Calculate the percentage change in price
            df['price_change_percentage'] = (
                df['future_close'] - df['close']) / df['close'] * 100

            # Define labels: 1 for long, 0 for do nothing, -1 for short, based on the threshold
            df['label'] = np.where(df['price_change_percentage'] > threshold, 1,
                                   np.where(df['price_change_percentage'] < -threshold, -1, 0))

            # Fill NaN values in 'label' column
            df['label'].fillna(0, inplace=True)

            # Save the 'label' column to a CSV file
            df['label'].to_csv('data/csv/labels.csv', index=False)

            self.logger.info(df['label'])
            return df['label']
        else:
            return df['close']

    def train_evaluate_and_save_model(self, model: str):
        # Define the path of the pickle file
        pickle_file_path = 'data/pickle/2020/3h_data.pkl'

        # Check if the pickle file exists
        if os.path.exists(pickle_file_path):
            # If it exists, load the data from the pickle file
            df = load(pickle_file_path)
        else:
            # If it does not exist, fetch and preprocess the data
            data = self.exchange.fetch_ohlcv(
                self.symbol, timeframe='30m', limit=5_000)
            df = pd.DataFrame(
                data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

            # Save the preprocessed data as a pickle file for future use
            with open(pickle_file_path, 'wb') as file:
                dump(df, file)

        features = self.process_features(df)
        labels = self.process_labels(df, model, n_candles=5)

        # Drop the last two rows as they won't have valid labels
        features = features[:-2]
        labels = labels[:-2]

        # Scale features and labels
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Save the fitted scaler for future use
        joblib.dump(scaler, 'data/scaler/scaler.p', protocol=4)

        # only needed when you use strings
        # label_scaler = LabelEncoder()
        # labels_scaled = label_scaler.fit_transform(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2)

        if model in column_1d:
            y_train = y_train.ravel()
            y_test = y_test.ravel()
        else:
            y_train = array_min2d(y_train)
            y_test = array_min2d(y_test)

        # # MLPCLASSEFIER
        # if model == "MLPClassifier":
        #     X_train = np.asarray(X_train, dtype=object)

        # Train, evaluate, and save the model
        trainer = MLModelTrainer(model, self.logger)
        trained_model, score = trainer.train(
            X_train, y_train, self.column_names)

        accuracy = self.evaluate_model(model, trained_model, X_test, y_test)

        self.save_model(trained_model, accuracy, score)
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

    def load_pretrained_scaler(self):
        return joblib.load('data/scaler/scaler.pkl')


class MLModelTrainer:
    def __init__(self, algorithm, logger) -> None:
        """
        Initialize the MLModelTrainer with the given algorithm and logger.

        :param algorithm: Algorithm used for model training.
        :param logger: Logger for logging progress and errors.
        """

        self.algorithm = algorithm
        self.logger = logger
        self.stop_loading = False

    def train(self, X, y, feature_names):
        """
        Train the model using the given features X and target y.

        :param X: Features for training.
        :param y: Target variable for training.
        """

        model, parameters = get_model(self.algorithm)
        if model in classifier:
            # Create a scorer from the custom scoring function using lambda to handle n_candles
            monetary_scorer = make_scorer(lambda y_true, y_pred: future_score_clas(
                y_true, y_pred, n_candles=5), greater_is_better=True)
        else:
            # Create a scorer from the custom scoring function using lambda to handle n_candles
            monetary_scorer = make_scorer(lambda y_true, y_pred: future_score_reg(
                y_true, y_pred, n_candles=5), greater_is_better=True)

        try:
            # Assuming parallelize_search is a method defined elsewhere in your class
            best_estimator = self.parallelize_search(
                model, parameters, X, y, monetary_scorer, feature_names)
            self.logger.info("Model training successful.")

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")

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
        return best_estimator, best_f1_score

    def parallelize_search(self, estimator, param_grid, X, y, scorer, feature_names):
        grid_search_estimator = GridSearchCV(
            estimator, param_grid, cv=5, scoring=scorer)
        random_search_estimator = RandomizedSearchCV(
            estimator, param_grid, cv=5, scoring=scorer)
        bayes_search_estimator = BayesSearchCV(
            estimator, param_grid, cv=5, scoring=scorer)

        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            try:
                grid_search = executor.submit(grid_search_estimator.fit, X, y)
                random_search = executor.submit(
                    random_search_estimator.fit, X, y)
                bayes_search = executor.submit(
                    bayes_search_estimator.fit, X, y)

                futures = [grid_search, random_search, bayes_search]
                for future in as_completed(futures):

                    # This will re-raise any exception that occurred during execution.
                    result = future.result()
                    results.append(result)

            except Exception as e:
                self.logger.error(
                    f"Exception occurred during model search: {e}")
            finally:
                for future in futures:
                    if not future.done():
                        future.cancel()

        if not results:
            raise ValueError("All searches failed!")

        best_estimator = max(results, key=lambda x: x.best_score_)
        self.logger.info(best_estimator)

        # Get the best model from the tuned estimator
        best_model = best_estimator.best_estimator_

        # Perform feature importance analysis
        feature_importances = None
        if feature_names is not None:
            try:
                if hasattr(best_model, 'feature_importances_'):
                    feature_importances = best_model.feature_importances_
                elif hasattr(best_model, 'coef_'):
                    feature_importances = best_model.coef_.flatten()
            except Exception as e:
                self.logger.error(
                    f"Error calculating feature importances: {e}")

        output_csv = 'data/csv/feature_importances.csv'
        # Save feature importances to a CSV file
        if feature_importances is not None and output_csv is not None:
            try:
                feature_importance_df = pd.DataFrame({
                    'model': best_model,
                    'Feature': feature_names,
                    'Importance': feature_importances
                })
                feature_importance_df.to_csv(output_csv, index=False)
                self.logger.info(f"Feature importances saved to {output_csv}")
            except Exception as e:
                self.logger.error(
                    f"Error saving feature importances to CSV: {e}")

        return best_estimator
