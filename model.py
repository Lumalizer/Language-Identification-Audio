import torch
import torch.nn as nn
from options import Options
from torchaudio.transforms import MelSpectrogram
from load_data import get_dataloaders
import logging
import os


class LanguageClassifier(nn.Module):
    def __init__(self, options):
        super(LanguageClassifier, self).__init__()
        logging.info("Initializing model...")

        self.options = options
        self.num_languages = 6 if options.use_all_languages else 2

        self.mel_spectogram_transform = MelSpectrogram(
            sample_rate=options.sample_rate,
            n_fft = 512, # Higher n_fft, higher frequency resolution you get.
            hop_length = 256, # Smaller hop_length, higher time resolution.
            n_mels = 40,
            f_max = options.sample_rate / 2, # the Nyquist frequency -> 0.5 of the sampling rate
            norm = 'slaney' 
            )

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(
            5, 5), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(
            5, 5), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(128, 128, kernel_size=(
            5, 5), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.lstm = nn.LSTM(input_size=2176, hidden_size=256,
                            num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(512, self.num_languages)

        self.to(options.device)
        logging.info(f"Model initialized on {options.device}")

    def forward(self, x):
        x = self.mel_spectogram_transform(x)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Reshape the tensor to feed into LSTM
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.fc(x)

        return x
    
def save_model_weights(model: nn.Module, options: Options, name="model_state_dict.pt"):
    if not os.path.exists(options.model_path):
        os.makedirs(options.model_path)
    torch.save(model.state_dict(), os.path.join(
        options.model_path, name))
    
    # NOTE: This fails. Problem may have to do with: "Script module creation requires that the model's operations be traceable, so certain types of operations may not be supported."
    # torch.jit.save(torch.jit.script(model), os.path.join(
    #     options.model_path, name))

def load_model_weights(model: nn.Module, options: Options, name="model_state_dict.pt"):
    model.load_state_dict(torch.load(os.path.join(
        options.model_path, name)))
    return model

def train_model(model, train_loader, options: Options):
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    # reduce the learning after 20 epochs by a factor of 10
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=5, gamma=0.5)

    loss_function = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(options.n_epochs):
        for batch_i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(options.device)
            labels = labels.to(options.device)
            predictions = model(data)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Training: Epoch {epoch} / {options.n_epochs} Batch {batch_i} / {len(train_loader)} Loss {loss.item()}", end="\r")
    print("\n")
    return model

def test_model(model, test_loader, options: Options):
    model.eval()
    test_acc = 0
    for data, labels in test_loader:
        data = data.to(options.device)
        labels = labels.to(options.device)
        predictions = model(data)
        accuracy = (torch.max(predictions, dim=-1, keepdim=True)
                    [1].flatten() == labels).sum() / len(labels)
        test_acc += accuracy.item()
    test_acc /= len(test_loader)
    return f"Test accuracy {test_acc}"

def build_model(options: Options, train_loader, test_loader):
    model = LanguageClassifier(options)
    train_model(model, train_loader, options)
    print(test_model(model, test_loader, options))
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    options = Options(use_all_languages=True)
    train_loader, test_loader = get_dataloaders(options)
    model = LanguageClassifier(2, options)
    #model = load_model_weights(model, options, "model3_state_dict.pt")
    train_model(model, train_loader, options)
    print(test_model(model, test_loader, options))
    save_model_weights(model, options, f"model_{model.num_languages}languages_state_dict.pt")
