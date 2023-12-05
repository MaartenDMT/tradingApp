from model.models import LoginModel
from presenters import Presenter
from view.views import LoginView


def test_credentials_checking():
    model = LoginModel()
    view = LoginView()
    app = Presenter(model, view)
    app.run()

    model.set_credentials("user", "pass")
    assert model.check_credentials() == True

    model.set_credentials("wrong", "credentials")
    assert model.check_credentials() == False
