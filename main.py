from options import Options
from load_data import load_data
from q1a_plot import q1a_plot
from model import build_model

options = Options(normalize=True)

X_train, y_train, X_test, y_test = load_data(options)
# q1a_plot(options, X_train)
build_model = build_model(options, X_train, y_train, X_test, y_test)
