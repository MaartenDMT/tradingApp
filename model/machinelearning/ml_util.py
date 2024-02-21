

# models for classification
classifier = ["Logistic Regression", "MLPClassifier", "BernoulliRBM", "Decision Tree Classifier", "Random Forest Classifier", "SVC",
              "Isolation Forest", "Gradient Boosting Classifier", "Extra Tree Classifier", "XGBoost Classifier",
              "Gaussian Naive Bayes", "Radius Neighbors Classifier", "K-Nearest Neighbors", "AdaBoost Classifier",
              "Gaussian Process Classifier", "Quadratic Discriminant Analysis", "SGD Classifier"]


# models needing 1d arrays
column_1d = ["Random Forest Classifier", "Gradient Boosting Classifier", "SVC",
             "Logistic Regression", "Decision Tree Classifier", "MLPClassifier", 'BernoulliRBM',
             "SVR", 'LinearSVR', 'NuSVR', "Extra Tree Classifier", "XGBoost Classifier",
             "Linear Regression", "Radius Neighbors Classifier", "Gaussian Naive Bayes",
             "K-Nearest Neighbors", "AdaBoost Classifier", "Gaussian Process Classifier",
             "Quadratic Discriminant Analysis", "SGD Classifier", "Ridge Regression",
             "Lasso Regression", "Elastic Net Regression",
             "Bayesian Ridge Regression", "SGD Regressor", 'Huber Regressor', 'Lars', 'RANSAC Regressor', 'Theil-Sen Regressor', "Gradient Boosting Regressor"]

# models for regression
regression = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression', 'Bayesian Ridge Regression',
              "Gradient Boosting Regressor", 'SVR', 'LinearSVR', 'NuSVR',
              'SGD Regressor', 'Huber Regressor', 'Lars', 'RANSAC Regressor', 'Theil-Sen Regressor']


tradex_features = ['ema55H', 'ema55H', 'ema_100', 'ema_200',
                   'lsma', 'ema_10', 'golden_signal', 'vwap', 'wma']


def future_score_clas(y_true, y_pred, n_candles=2, starting_capital=10000, transaction_cost_percent=0.01, trade_percent=0.1, stop_loss=0.05, take_profit=0.08):
    capital = starting_capital
    position = 1  # 0 for short, 1 for hold, 2 for long
    entry_price = 0

    for i in range(min(len(y_pred), len(y_true) - n_candles)):
        trade_amount = capital * trade_percent
        transaction_cost = trade_amount * transaction_cost_percent

        if position != 1:
            current_return = (y_true[i + n_candles] -
                              entry_price) * (1 if position == 2 else -1)
            if trade_amount > 0 and entry_price > 0:
                if current_return / (trade_amount * entry_price) <= -stop_loss or current_return / (trade_amount * entry_price) >= take_profit:
                    capital += current_return - transaction_cost
                    position = 1  # Reset position to hold

        if position == 1:
            entry_price = y_true[i]
            if y_pred[i] == 2:  # Go long
                position = 2
                capital -= transaction_cost
            elif y_pred[i] == 0:  # Go short
                position = 0
                capital -= transaction_cost

        if capital < starting_capital * 0.9:
            break  # Stop-loss trigger

    if position != 1:
        current_return = (y_true[-1] - entry_price) * \
            (1 if position == 2 else -1)
        capital += current_return - transaction_cost

    return_percentage = ((capital - starting_capital) / starting_capital) * 100
    return return_percentage


def future_score_reg(y_true, y_pred, n_candles=2, starting_capital=10000, threshold_percentage=0.07, transaction_cost_percent=0.01, trade_percent=0.1, stop_loss=0.05, take_profit=0.1):
    capital = starting_capital
    position = 1  # 0 for short, 1 for hold, 2 for long
    entry_price = 0

    for i in range(min(len(y_pred), len(y_true) - n_candles)):
        trade_amount = capital * trade_percent
        transaction_cost = trade_amount * transaction_cost_percent

        if position != 1:
            current_return = (y_true[i + n_candles] -
                              entry_price) * (1 if position == 2 else -1)

            if trade_amount > 0 and entry_price > 0:
                if current_return / (trade_amount * entry_price) <= -stop_loss or current_return / (trade_amount * entry_price) >= take_profit:
                    capital += current_return - transaction_cost
                    position = 1  # Reset position to hold

        predicted_return = y_pred[i] - y_true[i]
        threshold = threshold_percentage * y_true[i]

        if position == 1:
            if predicted_return > threshold:  # Go long
                position = 2
                entry_price = y_true[i]
                capital -= transaction_cost
            elif predicted_return < -threshold:  # Go short
                position = 0
                entry_price = y_true[i]
                capital -= transaction_cost

        if capital < starting_capital * 0.9:
            break  # Stop-loss trigger

    if position != 1:
        current_return = (y_true[-1] - entry_price) * \
            (1 if position == 2 else -1)
        capital += current_return - transaction_cost

    return_percentage = ((capital - starting_capital) / starting_capital) * 100
    return return_percentage


def spot_score_clas(y_true, y_pred, n_candles=2, starting_capital=10000, trade_fraction=0.1, transaction_cost_percent=0.01, stop_loss=0.05, take_profit=0.08):
    capital = starting_capital
    holding_stock = False
    buy_price = 0

    for i in range(min(len(y_pred), len(y_true) - n_candles)):
        if holding_stock:
            current_return = (y_true[i + n_candles] - buy_price) * trade_amount
            transaction_cost = trade_amount * transaction_cost_percent
            if current_return / buy_price <= -stop_loss or current_return / buy_price >= take_profit:
                capital += current_return - transaction_cost
                holding_stock = False

        if not holding_stock and y_pred[i] == 2:
            trade_amount = capital * trade_fraction
            transaction_cost = trade_amount * transaction_cost_percent
            buy_price = y_true[i]
            holding_stock = True
            capital -= transaction_cost

    return capital - starting_capital


def spot_score_reg(y_true, y_pred, n_candles=2, starting_capital=10000, trade_fraction=0.1, transaction_cost_percent=0.01, stop_loss=0.05, take_profit=0.08):
    capital = starting_capital
    holding_stock = False
    buy_price = 0

    for i in range(min(len(y_pred), len(y_true) - n_candles)):
        if holding_stock:
            current_return = (y_true[i + n_candles] - buy_price) * trade_amount
            transaction_cost = trade_amount * transaction_cost_percent
            if current_return / buy_price <= -stop_loss or current_return / buy_price >= take_profit:
                capital += current_return - transaction_cost
                holding_stock = False

        predicted_return = y_pred[i] - y_true[i]
        if not holding_stock and predicted_return > 0:
            trade_amount = capital * trade_fraction
            transaction_cost = trade_amount * transaction_cost_percent
            buy_price = y_true[i]
            holding_stock = True
            capital -= transaction_cost

    return capital - starting_capital

# data/ml/plots/shap/shap_summary_plot.png


"""
Logistic Regression
MLPClassifier (Multi-layer Perceptron classifier)
Decision Tree Classifier
Random Forest Classifier
SVC (Support Vector Classifier)
Isolation Forest (More commonly used for anomaly detection)
Gradient Boosting Classifier
Extra Tree Classifier
XGBoost Classifier
Gaussian Naive Bayes
Radius Neighbors Classifier
K-Nearest Neighbors
AdaBoost Classifier
Gaussian Process Classifier
Quadratic Discriminant Analysis
SGD Classifier (Stochastic Gradient Descent Classifier)


Linear Regression: A fundamental regression technique for modeling the relationship 
between a scalar response and one or more explanatory variables.

SVR (Support Vector Regression): An extension of Support Vector Machines (SVM) that 
supports regression tasks.

Ridge Regression: A technique for analyzing multiple regression data that suffer from 
multicollinearity. It applies L2 regularization.

Lasso Regression: Similar to Ridge Regression but uses L1 regularization, which can lead 
to sparse models with few coefficients.

Elastic Net Regression: Combines L1 and L2 priors (regularization terms) as regularizer 
in linear regression models.

Decision Tree Regressor: Uses a decision tree for regression. It's a tree-structured 
model used to predict continuous values.

Bayesian Ridge Regression: Implements Bayesian ridge regression, which includes 
regularization parameters that are estimated from the data.

Gradient Boosting Regressor: A machine learning technique for regression problems, 
which builds a model in a stage-wise fashion like other boosting methods do, but it 
generalizes them by allowing optimization of an arbitrary differentiable loss function.

SGD Regressor (Stochastic Gradient Descent Regressor): Fits a linear model using 
stochastic gradient descent. It's useful for large-scale and sparse data.
"""
