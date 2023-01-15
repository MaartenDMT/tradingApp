from tkinter import Frame

from ttkbootstrap import Button, Label, OptionMenu, StringVar


class MLTab(Frame):
    def __init__(self, parent, presenter) ->None:
        super().__init__(parent)
        self.parent = parent
        
        self._presenter = presenter
        # ... GUI elements for the trade page go here ...
        
        # select the ML algo 
        self.select_ml_label = Label(
        self, text='Select a ML Algorithm', font=('Arial', 12))
        
        self.type_var = StringVar(self)
        self.type_var.set("Linear Regression")
        self.ml_select_menu = OptionMenu(
            self, self.type_var, "Linear Regression", "Logistic Regression", 
            "MLPClassifier", "Decision Tree Classifier", "Random Forest Classifier", "SVC", "SVR",
            "Isolation Forest", "Gradient Boosting Classifier", "Extra Tree Classifier")
        self.select_ml_button = Button(self, text="Select ML", 
                                       command=self._presenter.get_ML_model)
        
        # the selected model
        self.show_ml_label = Label(
        self, text='Current select Algorithm', font=('Arial', 12))
        self.current_ml_label = Label(
        self, text='-', font=('Arial', 10))
        
        #train evaluate en save the model
        self.evaluate_save_label = Label(
        self, text='train evaluate en save the selected model', font=('Arial', 12))
        self.evaluate_save_button = Button(self, text="Select ML", 
                                           command=self._presenter.train_evaluate_save_model)
        
        # Put the elements on the grid
        self.select_ml_label.grid(row=0, column=0)
        self.ml_select_menu.grid(row=1, column=0)
        self.select_ml_button.grid(row=3, column=0)
        
        self.show_ml_label.grid(row=0, column=2)
        self.current_ml_label.grid(row=1, column=2)
        
        
        self.evaluate_save_label.grid(row=5, column=0, columnspan=2)
        self.evaluate_save_button.grid(row=6, column=0, columnspan=2)
        
