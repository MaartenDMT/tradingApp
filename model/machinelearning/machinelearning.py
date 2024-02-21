import datetime
import os
import threading
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from joblib import dump, load
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, explained_variance_score,
                             make_scorer, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

import util.loggers as loggers
from model.machinelearning.ml_models import get_model
from model.machinelearning.ml_util import (classifier, column_1d,
                                           future_score_clas, future_score_reg,
                                           regression, spot_score_clas,
                                           spot_score_reg)
from util.utils import array_min2d, tradex_features

# Disable ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Use the 'Agg' backend
matplotlib.use('Agg')
# import pandas_ta as ta

logger = loggers.setup_loggers()

PLOTS_PATH = 'data/ml/plots'
MODEL_PERFO_PATH = 'data/ml/csv/model_performance.csv'
SHIFT_CANDLES = 3


class MachineLearning:
    def __init__(self, exchange, symbol) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.logger = logger['model']
        self.shap_interpreter = SHAPClass()

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
            return 1  # return a default value or handle according to your use case

    def evaluate_model(self, name, model, X_test, y_test):
        y_pred = model.predict(X_test)

        if name in regression:
            accuracy = self.evaluate_regression_model(name, y_test, y_pred)
        else:
            accuracy = self.evaluate_classification_model(name, y_test, y_pred)

        # Use the interpret_model method to get SHAP values and plots
        _ = self.interpret_model(
            model, X_test, save_values=True, plot_values=True, feature_names=self.column_names)

        return accuracy

    def evaluate_regression_model(self, name, y_test, y_pred):
        # Calculate and log regression metrics
        r2, mae, mse, rmse, evs = self.calculate_regression_metrics(
            y_test, y_pred)

        # Plot regression results
        self.plot_regression_results(
            name, y_test, y_pred, r2, mae, mse, rmse, evs)

        return r2

    def evaluate_classification_model(self, name, y_test, y_pred):
        # Calculate and log classification metrics
        accuracy, auc = self.calculate_classification_metrics(y_test, y_pred)

        # Plot classification results
        self.plot_classification_results(name, y_test, y_pred, accuracy, auc)

        return accuracy

    def calculate_regression_metrics(self, y_test, y_pred):
        # Compute various regression metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        evs = explained_variance_score(y_test, y_pred)

        # Log the metrics
        self.logger.info(f"R2 Score: {r2}")
        self.logger.info(f"Mean Absolute Error: {mae}")
        self.logger.info(f"Mean Squared Error: {mse}")
        self.logger.info(f"Root Mean Squared Error: {rmse}")
        self.logger.info(f"Explained Variance Score: {evs}")

        return r2, mae, mse, rmse, evs

    def calculate_classification_metrics(self, y_test, y_pred):
        # Compute various classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred) if len(
            set(y_test)) == 2 else "Not Applicable"
        self.logger.info(f"Accuracy: {accuracy}")
        self.logger.info(f"AUC-ROC: {auc}")
        self.logger.info(
            f"Classification Report:\n{classification_report(y_test, y_pred)}")

        return accuracy, auc

    def plot_regression_results(self, name, y_test, y_pred, r2, mae, mse, rmse, evs):
        plot_dir = PLOTS_PATH
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot for true values vs predictions
        ax.scatter(y_test, y_pred)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f'Regression Results for {name} - True vs Predicted')

        # Plotting the line of best fit
        ax.plot([y_test.min(), y_test.max()], [
                y_test.min(), y_test.max()], 'k--', lw=4)

        # Adding text for metrics
        textstr = f'R2: {r2:.2f}\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nEVS: {evs:.2f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save and show the plot
        plt.savefig(f'{plot_dir}/{name}_regression_plot.png')

        # Show or close the plot
        if threading.current_thread() == threading.main_thread():
            plt.show()
        else:
            plt.close()

    def plot_classification_results(self, name, y_test, y_pred, accuracy, auc_score):
        plot_dir = PLOTS_PATH
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix for {name}')

        # Plot ROC curve if binary classification
        if len(set(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
            ax_roc.plot(fpr, tpr, color='blue', lw=2,
                        label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'ROC Curve for {name}')
            ax_roc.legend(loc="lower right")
            plt.savefig(f'{plot_dir}/{name}_roc_curve.png')

        # Save and show the plots
        plt.savefig(f'{plot_dir}/{name}_confusion_matrix.png')

        # Show or close the plot
        if threading.current_thread() == threading.main_thread():
            plt.show()
        else:
            plt.close()

    def save_model(self, model, accuracy, score=None):
        path = r'data/ml/2020/'
        params_path = r'data/ml/csv/params_accuracy.csv'

        # Check if the best estimator is a Pipeline and extract the actual model
        if isinstance(model.best_estimator_, Pipeline):
            actual_model = model.best_estimator_.named_steps['regression']
        else:
            actual_model = model.best_estimator_

        if accuracy > 0.60:
            version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            model_name = actual_model.__class__.__name__
            model_filename = f'{path}trained_{model_name}_{version}_{accuracy*100:.2f}.p'

            joblib.dump(actual_model, model_filename)
            self.logger.info(f'Model saved as {model_filename}')

            model_params_accuracy = {
                "Model": model_name,
                "Version": version,
                "Params": str(model.best_params_),
                "Accuracy": accuracy,
                "Money": score
            }

            if os.path.exists(params_path):
                df = pd.read_csv(params_path)
                new_df = pd.DataFrame([model_params_accuracy])
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = pd.DataFrame([model_params_accuracy])

            df.to_csv(params_path, index=False)

            self.log_model_performance(actual_model, accuracy, score)

    def load_model(self, filename):
        path = r'data/ml/2020'
        file = f'{path}/{filename}'
        model = None  # Initialize model with a default value
        try:
            # Open the file and load the model
            model = joblib.load(file)
            self.logger.info(f'the model is : {model}')
        except FileNotFoundError:
            self.logger.error(f'there is no file called: {file}')

        return model

    def process_features(self, df):

        processed_features = tradex_features(self.symbol, df)

        self.column_names = processed_features.columns.tolist()

        # Fill missing values in the entire DataFrame
        processed_features = processed_features.fillna(0)

        # Return the processed features as a NumPy array
        return processed_features.to_numpy()

    def process_labels(self, df, model, n_candles=3, threshold=0.08):
        if model in classifier:
            # Shift closing prices by -2 to compare with the price two candles ahead
            df['future_close'] = df['close'].shift(-n_candles)

            # Calculate the percentage change in price
            df['price_change_percentage'] = (
                df['future_close'] - df['close']) / df['close'] * 100

            # Define labels: 2 for long, 1 for do nothing, 0 for short, based on the threshold
            df['label'] = np.where(df['price_change_percentage'] >= threshold, 2,
                                   np.where(df['price_change_percentage'] <= -threshold, 0, 1))

            # Fill NaN values in 'label' column
            df['label'].fillna(1, inplace=True)

            # Save the 'label' column to a CSV file
            df[['label', 'price_change_percentage', 'future_close', 'close']].to_csv(
                'data/ml/csv/labels.csv', index=False)

            self.logger.info(df['label'])
            return df['label']
        else:
            return df['close']

    def train_evaluate_and_save_model(self, model: str, usage_percent=0.7):
        # Define the path of the pickle file
        pickle_file_path = 'data/2020/30m_data.pkl'
        csv_file_path = 'data/csv/BTC_1h.csv'

        # Initialize the DataFrame
        df = None

        # Check if the CSV file exists
        if os.path.exists(csv_file_path):
            # If it exists, load the data from the CSV file
            df = pd.read_csv(csv_file_path)
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            self.logger.info("CSV is loaded")
        # elif os.path.exists(pickle_file_path):
        #     # If the CSV file doesn't exist, but the pickle file does, load the pickle file
        #     df = pd.read_pickle(pickle_file_path)
        #     df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        #     self.logger.info("PICKLE is loaded")
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

        # Adjust the DataFrame to use only the specified percentage of the data
        split_index = int(len(df) * usage_percent)
        df = df.iloc[:split_index]

        self.logger.info(
            f"Using {len(df)} out of {len(df) / (usage_percent)} rows ({usage_percent}%) of the data.")

        features = self.process_features(df)
        labels = self.process_labels(df, model, n_candles=SHIFT_CANDLES)

        # Drop the last two rows as they won't have valid labels
        features = features[:-SHIFT_CANDLES]
        labels = labels[:-SHIFT_CANDLES]

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

    def log_model_performance(self, model, accuracy, score):
        model_performance_file = MODEL_PERFO_PATH
        model_info = {
            "Model": model.__class__.__name__,
            "Version": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            "Accuracy": accuracy,
            "Score": score,
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Convert the dictionary to a DataFrame
        model_info_df = pd.DataFrame([model_info])

        # Check if file exists to append or write new
        if os.path.isfile(model_performance_file):
            model_info_df.to_csv(model_performance_file,
                                 mode='a', header=False, index=False)
        else:
            model_info_df.to_csv(model_performance_file,
                                 mode='w', header=True, index=False)

        self.compare_model_performance()

    def compare_model_performance(self):
        model_performance_file = MODEL_PERFO_PATH
        if not os.path.isfile(model_performance_file):
            self.logger.info("No model performance data found.")
            return

        df = pd.read_csv(model_performance_file)
        # Implement your logic to compare models here
        # For example, you can sort by accuracy, display top N models, etc.
        sorted_df = df.sort_values(by=['Accuracy'], ascending=False)
        self.logger.info(sorted_df.head())  # Display top models

    def interpret_model(self, model, X_test, save_values=False, plot_values=False, feature_names=None):
        """
        Interpret the model predictions using SHAP values.

        :param model: The trained machine learning model.
        :param X_test: Test dataset (features) used for the model.
        """
        shap_values = self.shap_interpreter.explain_prediction(
            model, X_test, save_values, plot_values, feature_names)
        # Additional code to handle or display SHAP values can be added here
        return shap_values


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
        self.executor = ThreadPoolExecutor()
        self.futures = []

    def train(self, X, y, feature_names):
        """
        Train the model using the given features X and target y.

        :param X: Features for training.
        :param y: Target variable for training.
        """
        best_estimator = None  # Initialize to None
        model, parameters = get_model(self.algorithm)
        try:
            if self.algorithm in classifier:
                # Create a scorer from the custom scoring function using lambda to handle n_candles
                monetary_scorer = make_scorer(lambda y_true, y_pred: future_score_clas(
                    y_true, y_pred, n_candles=SHIFT_CANDLES), greater_is_better=True)
            elif self.algorithm in regression:
                # Create a scorer from the custom scoring function using lambda to handle n_candles
                monetary_scorer = make_scorer(lambda y_true, y_pred: future_score_reg(
                    y_true, y_pred, n_candles=SHIFT_CANDLES), greater_is_better=True)
            else:
                raise ValueError("No Model Found.")
        except Exception as e:
            self.logger.error(
                f"error with the make scorer method: {e}\n{traceback.format_exc()}")

        try:
            # Assuming parallelize_search is a method defined elsewhere in your class
            best_estimator = self.parallelize_search(
                model, parameters, X, y, monetary_scorer, feature_names)
            self.logger.info("Model training successful.")

        except Exception as e:
            self.logger.error(
                f"Model training failed: {e}\n{traceback.format_exc()}")
            # Handle the case when best_estimator is not assigned
            if best_estimator is None:
                # Handle appropriately, maybe return None or raise an exception
                raise ValueError(
                    "Model training failed and no best_estimator found.")
            return None, None  # Return None or handle the exception as needed

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
        # grid_search_estimator = GridSearchCV(
        #     estimator, param_grid, cv=5, scoring=scorer, verbose=2, n_jobs=8)
        random_search_estimator = RandomizedSearchCV(
            estimator, param_grid, cv=5, scoring=scorer, verbose=2, n_jobs=6)
        bayes_search_estimator = BayesSearchCV(
            estimator, param_grid, cv=5, scoring=scorer, verbose=2, n_jobs=6, n_points=4)

        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            try:
                # grid_search = executor.submit(grid_search_estimator.fit, X, y)
                random_search = executor.submit(
                    random_search_estimator.fit, X, y)
                bayes_search = executor.submit(
                    bayes_search_estimator.fit, X, y)

                futures = [random_search, bayes_search]
                for future in as_completed(futures):

                    # This will re-raise any exception that occurred during execution.
                    result = future.result()
                    results.append(result)

            except Exception as e:
                self.logger.error(
                    f"Exception occurred during model search: {e}")
                self.shutdown_thread_pool()
            finally:
                self.shutdown_thread_pool()

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

        output_csv = 'data/ml/csv/feature_importances.csv'

        # Create new DataFrame for feature importances
        if feature_importances is not None and output_csv is not None:
            try:
                # Create a dictionary for the new DataFrame
                data_dict = {
                    'model': [str(best_model)],
                    'best_score': [best_estimator.best_score_]
                }
                for i, feature_name in enumerate(feature_names):
                    data_dict[feature_name] = [feature_importances[i]]

                # Create DataFrame with new data
                new_feature_importance_df = pd.DataFrame(data_dict)

                # Check if CSV file exists
                if os.path.exists(output_csv):
                    # Read existing data
                    existing_df = pd.read_csv(output_csv)
                    # Append new data
                    updated_df = pd.concat(
                        [existing_df, new_feature_importance_df], ignore_index=True)
                else:
                    # Use new data as the DataFrame
                    updated_df = new_feature_importance_df

                # Save the updated DataFrame
                updated_df.to_csv(output_csv, index=False)
                self.logger.info(
                    f"Feature importances and best score updated in {output_csv}")
            except Exception as e:
                self.logger.error(
                    f"Error updating feature importances and best score in CSV: {e}")

        return best_estimator

    def shutdown_thread_pool(self):
        # Cancel all futures that are not yet running
        for future in self.futures:
            future.cancel()

        # Shutdown the executor
        self.executor.shutdown(wait=True)
        self.logger.info("Thread pool shut down successfully.")


class SHAPClass:
    def __init__(self) -> None:
        self.logger = logger['model']

    def explain_prediction(self, model, feature_array, save_values=False, plot_values=False, feature_names=None):
        """
        Generate SHAP values for a given model and feature array, 
        with options to save and plot the values.

        :param model: The trained machine learning model.
        :param feature_array: A numpy array or pandas DataFrame of feature values.
        :param save_values: Boolean flag to save SHAP values.
        :param plot_values: Boolean flag to plot SHAP values.
        :return: SHAP values.
        """

        # Check if the model is a GridSearchCV instance
        if isinstance(model, (GridSearchCV, BayesSearchCV, RandomizedSearchCV)):
            model = model.best_estimator_

        if isinstance(model, Pipeline):
            model = model.named_steps['regression']
        # Debugging: Print the type of the model and test the predict method
        self.logger.info(f"Model type: {type(model)}")
        try:
            sample_prediction = model.predict(feature_array[:1])
            self.logger.info(feature_array[:1])
            self.logger.info(f"Sample Prediction: {sample_prediction}")
        except Exception as e:
            self.logger.info(f"Error making sample prediction: {e}")
            return None

        # Initialize the SHAP explainer with the model
        try:
            explainer = shap.Explainer(model, feature_names=feature_names)
        except Exception as e:
            self.logger.info(
                f"Error shap explainer: {e}\n{traceback.format_exc()}")
            return None
        # Calculate SHAP values
        shap_values = explainer.shap_values(feature_array)

        # Optionally save SHAP values
        if save_values:
            self.save_shap_values(
                shap_values, 'data/ml/json/shap/shap_values.json')

        # Optionally plot SHAP values
        if plot_values:
            self.plot_shap_values(shap_values, feature_array, feature_names)

        return shap_values

    def plot_shap_values(self, shap_values, features, feature_names):
        """
        Plot SHAP values for a given set of features.

        :param shap_values: The SHAP values.
        :param features: The features corresponding to the SHAP values.
        """
        shap.summary_plot(shap_values, features, show=False,
                          feature_names=feature_names)

        # Save the plot
        plt.savefig('data/ml/plots/shap/shap_summary_plot.png')
        self.logger.info("SHAP summary plot saved as 'shap_summary_plot.png'")

    def save_shap_values(self, shap_values, filename):
        """
        Save SHAP values to a file.

        :param shap_values: The SHAP values to save.
        :param filename: The filename to save the SHAP values to.
        """
        try:
            with open(filename, 'wb') as file:  # Open file in binary mode
                joblib.dump(shap_values, file)
            self.logger.info(f"SHAP values saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving SHAP values: {e}")
