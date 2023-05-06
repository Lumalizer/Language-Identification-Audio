import torch
import torch.nn as nn
from options import Options
from torchaudio.transforms import MelSpectrogram
from load_data import get_english_spanish_dataloaders
from model import test_model, train_model, save_model_weights, load_model_weights
import logging
import os


class LanguageClassifier(nn.Module):
    def __init__(self, num_languages, options):
        super(LanguageClassifier, self).__init__()

        logging.info("Initializing model...")

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

        self.fc = nn.Linear(512, num_languages)

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


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    options = Options()

    train_loader, test_loader = get_english_spanish_dataloaders(options)

    model = LanguageClassifier(2, options)

    model = load_model_weights(model, options, "model3_state_dict.pt")
    #train_model(model, train_loader, options)
    print(test_model(model, test_loader, options))
    #save_model_weights(model, options, "model3_state_dict.pt")
