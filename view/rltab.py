from tkinter import Frame

from ttkbootstrap import Button, Label


class RLTab(Frame):
    def __init__(self, parent, presenter) -> None:
        super().__init__(parent)
        self.parent = parent
        self._presenter = presenter
        # ... GUI elements for the trade page go here ...

        # select the ML algo
        self.select_ml_label = Label(
            self, text='Welcome to the Reinforcement Tab', font=('Arial', 12))

        # train evaluate en save the model
        self.evaluate_save_label = Label(
            self, text='train evaluate en save the selected model', font=('Arial', 12))
        self.evaluate_save_button = Button(self, text="train RL",
                                           command=self._presenter.rl_tab.train_evaluate_save_rlmodel)

        self.start_button = Button(self, text="start RL",
                                   command=self._presenter.rl_tab.start_rlmodel)

        # Put the elements on the grid
        self.select_ml_label.grid(row=0, column=0)

        self.evaluate_save_label.grid(row=1, column=0, columnspan=2)
        self.evaluate_save_button.grid(row=1, column=2, columnspan=2)
        self.start_button.grid(row=2, column=0, columnspan=2)
