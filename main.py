from options import Options
from plot import q1a_plot, q3a_plot
from model import build_model, save_model_weights
from load_data import get_dataloaders
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

options = Options(use_all_languages=True)
print(options)

train_loader, test_loader = get_dataloaders(options)

# q1a_plot(options, X_train)
model, train_losses, test_losses = build_model(options, train_loader, test_loader)
# q3a_plot(options, train_losses, test_losses)
save_model_weights(model, options)
