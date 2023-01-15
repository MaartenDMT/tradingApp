from model.models import Models
from presenters import Presenter
from view.views import WindowView

if __name__ == "__main__":
    model = Models()
    view = WindowView()
    app = Presenter(model, view)
    app.run()