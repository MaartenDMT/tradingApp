

# models for classification
classifier = ["Logistic Regression", "MLPClassifier", "Decision Tree Classifier", "Random Forest Classifier", "SVC",
              "Isolation Forest", "Gradient Boosting Classifier", "Extra Tree Classifier", "XGBoost Classifier",
              "Gaussian Naive Bayes", "Radius Neighbors Classifier", "K-Nearest Neighbors", "AdaBoost Classifier",
              "Gradient Boosting Regressor", "Gaussian Process Classifier", "Quadratic Discriminant Analysis", "SGD Classifier"]


# models needing 1d arrays
column_1d = ["Random Forest Classifier", "Gradient Boosting Classifier", "SVC",
             "Logistic Regression", "Decision Tree Classifier", "MLPClassifier",
             "SVR", "Extra Tree Classifier", "XGBoost Classifier",
             "Linear Regression", "Radius Neighbors Classifier", "Gaussian Naive Bayes",
             "K-Nearest Neighbors", "AdaBoost Classifier", "Gaussian Process Classifier",
             "Quadratic Discriminant Analysis", "SGD Classifier", "Ridge Regression",
             "Lasso Regression", "Elastic Net Regression", "Decision Tree Regressor",
             "Bayesian Ridge Regression", "SGD Regressor", "Gradient Boosting Regressor"]

# models for regression
regression = ['Linear Regression', 'SVR', 'Ridge Regression', 'Lasso Regression',
              'Elastic Net Regression', 'Decision Tree Regressor', 'Bayesian Ridge Regression', 'SGD Regressor']


tradex_features = ['ema55H', 'ema55H', 'ema_100', 'ema_200',
                   'lsma', 'ema_10', 'golden_signal', 'vwap', 'wma']


def future_score_clas(y_true, y_pred, n_candles=2, starting_capital=10000, threshold_percentage=0.3):
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

        # Calculate the return as a percentage of the starting capital
    return_percentage = (total_return / starting_capital) * 100

    return return_percentage


def future_score_reg(y_true, y_pred, n_candles=2, starting_capital=10000, threshold_percentage=0.1):
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

        predicted_return = y_pred[i] - y_true[i]
        threshold = threshold_percentage * y_true[i]
        if predicted_return > threshold:  # Predicted price increase -> go long
            entry_price = y_true[i]
            position = 1
        elif predicted_return < -threshold:  # Predicted price decrease -> go short
            entry_price = y_true[i]
            position = -1

    # Check if we are holding any position at the end
    if position != 0:
        total_return += (y_true[-1] - entry_price) * position

    # Calculate the return as a percentage of the starting capital
    return_percentage = (total_return / starting_capital) * 100

    return return_percentage


def spot_score_clas(y_true, y_pred, n_candles=2):
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


def spot_score_reg(y_true, y_pred, n_candles=2):
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

        predicted_return = y_pred[i] - y_true[i]
        if predicted_return > 0:  # Predicted price increase -> buy
            buy_price = y_true[i]
            holding_stock = True
        # No need for else condition as we do not buy if the predicted return is not positive

    # Check if we are holding any stock at the end
    if holding_stock:
        total_return += y_true[-1] - buy_price

    return total_return
