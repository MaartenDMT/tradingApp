from matplotlib import use
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ttkbootstrap import Button, Frame

use('TkAgg')


class ChartTab(Frame):
    def __init__(self, parent, presenter) -> None:
        super().__init__(parent)
        self.parent = parent
        self._presenter = presenter

        # Create a figure and a canvas to display the chart
        # Add a button for enabling and disabling automatic charting
        self.auto_chart = False
        self.start_autochart_button = Button(
            self, text="Start Auto Charting", command=self._presenter.chart_tab.toggle_auto_charting)
        self.figure = Figure(figsize=(4, 3), dpi=80)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.draw()
        self.axes.set_ylabel('Price')
        self.axes.set_xlabel('Time')

        # chart PAGE
        self.start_autochart_button.pack(
            side='bottom', fill='none', expand=0, padx='5', pady='5')
        self.canvas.get_tk_widget().pack(side='left', fill='both',
                                         expand=1, padx='5', pady='5')
