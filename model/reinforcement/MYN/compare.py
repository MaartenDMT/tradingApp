from agent import DQLAgent


def compare_models(X_train, y_train, X_test, y_test):
    agent = DQLAgent()

    dense_accuracy = agent.train_base_dense_model(
        X_train, y_train, X_test, y_test)
    lstm_accuracy = agent.train_base_lstm_model(
        X_train, y_train, X_test, y_test)
    conv_lstm_accuracy = agent.train_base_conv1d_lstm_model(
        X_train, y_train, X_test, y_test)

    best_model = max([
        ("Dense", dense_accuracy),
        ("LSTM", lstm_accuracy),
        ("Conv1D LSTM", conv_lstm_accuracy)
    ], key=lambda x: x[1])

    print(
        f"The best performing model is {best_model[0]} with an accuracy of {best_model[1]}.")

# Call compare_models with your actual data
# compare_models(X_train, y_train, X_test, y_test)
