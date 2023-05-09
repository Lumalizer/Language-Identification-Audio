from options import Options
from q1a_plot import q1a_plot
from model import build_model, save_model_weights
from load_data import get_dataloaders

options = Options(use_all_languages=True)
print(options)

train_loader, test_loader = get_dataloaders(options)

# q1a_plot(options, X_train)
model = build_model(options, train_loader, test_loader)
save_model_weights(model, options)
