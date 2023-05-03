from options import Options
from load_data import load_data
from q1a_plot import q1a_plot
from model import build_model
from load_data import convert_tensors

options = Options(normalize=True)


X_train, y_train, X_test, y_test = load_data(options)
train_loader, test_loader = convert_tensors(X_train, y_train, X_test, y_test)

# q1a_plot(options, X_train)
model = build_model(options, X_train, y_train, X_test, y_test)

