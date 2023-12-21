from tkinter import Frame

from ttkbootstrap import Button, Label, OptionMenu, StringVar


class MLTab(Frame):
    def __init__(self, parent, presenter) -> None:
        super().__init__(parent)
        self.parent = parent
        self._presenter = presenter

        # Create labels and set initial values
        self.select_ml_label = Label(
            self, text='Select a Machine Learning Algorithm', font=('Arial', 12))

        self.type_var = StringVar(self)
        self.type_var.set("Linear Regression")
        self.ml_select_menu = OptionMenu(
            self, self.type_var, "Linear Regression", "SVR", "Ridge Regression", "Lasso Regression",
            "Elastic Net Regression", "Decision Tree Regressor", "Bayesian Ridge Regression", "SGD Regressor",
            "Logistic Regression", "MLPClassifier", "Decision Tree Classifier", "Random Forest Classifier", "SVC", "Isolation Forest",
            "Gradient Boosting Classifier", "Extra Tree Classifier", "XGBoost Classifier",
            "Gaussian Naive Bayes", "Radius Neighbors Classifier", "K-Nearest Neighbors", "AdaBoost Classifier",
            "Gradient Boosting Regressor", "Gaussian Process Classifier", "Quadratic Discriminant Analysis", "SGD Classifier",
        )
        self.select_ml_button = Button(self, text="Select ML Algorithm",
                                       command=self._presenter.ml_tab.get_ML_model)

        # Display the selected model
        self.show_ml_label = Label(
            self, text='Current Selected Algorithm', font=('Arial', 12))
        self.current_ml_label = Label(
            self, text=self.type_var.get(), font=('Arial', 10))

        # Train, evaluate, and save the model
        self.evaluate_save_label = Label(
            self, text='Train, Evaluate, and Save the Selected Model', font=('Arial', 12))
        self.evaluate_save_button = Button(self, text="Train and Save Model",
                                           command=self._presenter.ml_tab.train_evaluate_save_model)

        # Place the elements on the grid
        self.select_ml_label.grid(row=0, column=0, padx=10, pady=10)
        self.ml_select_menu.grid(row=1, column=0, padx=10, pady=10)
        self.select_ml_button.grid(row=2, column=0, padx=10, pady=10)

        self.show_ml_label.grid(row=0, column=1, padx=10, pady=10)
        self.current_ml_label.grid(row=1, column=1, padx=10, pady=10)

        self.evaluate_save_label.grid(
            row=3, column=0, columnspan=2, padx=10, pady=10)
        self.evaluate_save_button.grid(
            row=4, column=0, columnspan=2, padx=10, pady=10)
