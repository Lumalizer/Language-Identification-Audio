from options import Options
from load_data import load_data
from q1a_plot import q1a_plot


options = Options(normalize=True)

X_train, y_train, X_test, y_test = load_data(options)
q1a_plot(options, X_train)
