from options import Options
from q1a_plot import q1a_plot
from model import build_model
from load_data import get_dataloaders, get_english_spanish_dataloaders

options = Options()

train_loader, test_loader = get_english_spanish_dataloaders(options)

# q1a_plot(options, X_train)
model = build_model(options, train_loader, test_loader)
model.save()
