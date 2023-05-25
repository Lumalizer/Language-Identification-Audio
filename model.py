import torch
import torch.nn as nn
from options import Options
from torchaudio.transforms import MelSpectrogram, MFCC
from load_data import get_dataloaders
import logging
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np


class LanguageClassifier(nn.Module):
    def __init__(self, options: Options):
        super(LanguageClassifier, self).__init__()
        logging.info("Initializing model...")

        self.options = options
        self.num_languages = 6 if options.use_all_languages else 2

        self.lstm_hidden_size = 312

        self.mel_spectogram_transform = MelSpectrogram(
            sample_rate=options.sample_rate,
            n_fft=1024,  # Higher n_fft, higher frequency resolution you get.
            hop_length=256,  # Smaller hop_length, higher time resolution.
            n_mels=40,
            f_max=options.sample_rate / 2,  # the Nyquist frequency -> 0.5 of the sampling rate
            norm='slaney'
        )

        self.MFCC_transform = MFCC(
            sample_rate=options.sample_rate,
            n_mfcc=40,
            log_mels=True,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40,
                       "f_max": options.sample_rate / 2},
            norm='ortho'
        )

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.lstm = nn.LSTM(input_size=1152, hidden_size=self.lstm_hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(self.lstm_hidden_size * 2, self.num_languages)

        self.to(options.device)
        logging.info(f"Model initialized on {options.device}")

    def forward(self, x: torch.Tensor):
        # torch no_grad is used since we don't want to save the gradient
        # for the pre-processing steps
        with torch.no_grad():
            x = F.normalize(x, p=2.0, dim=1)  # L2 normalization
            # x = self.mel_spectogram_transform(x)
            x = self.MFCC_transform(x)

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

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Reshape the tensor to feed into LSTM
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        x, _ = self.lstm(x)

        x = x[:, -1, :]
        # x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def save_model_weights(model: nn.Module, options: Options, name="model_state_dict.pt"):
    if not os.path.exists(options.model_path):
        os.makedirs(options.model_path)
    # torch.save(model.state_dict(), os.path.join(
    #     options.model_path, name))

    path = os.path.join(options.model_path, f"JIT_{name}")
    torch.jit.save(torch.jit.script(model), path)

    logging.info(f"Model saved at {path}")


def load_model_weights(model: nn.Module, options: Options, name="model_state_dict.pt"):
    model.load_state_dict(torch.load(os.path.join(
        options.model_path, name)))
    return model


def load_jit_model(options: Options, name="JIT_model_state_dict.pt"):
    logging.info(
        f"Loading JIT model from {os.path.join(options.model_path, name)}")
    model = torch.jit.load(os.path.join(
        options.model_path, name))
    model = model.to(options.device)
    logging.info(f"Model loaded on {options.device}")
    return model


def train_model(model: LanguageClassifier, train_loader, loss_function, options: Options):
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=options.n_epochs // 1.5)

    history_epoch = np.zeros(options.n_epochs)
    history_batch = []

    model.train()
    print("Training:")
    for epoch in range(options.n_epochs):
        for batch_i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(options.device)
            labels = labels.to(options.device)

            predictions = model(data)
            loss = loss_function(predictions, labels)

            history_epoch[epoch] = loss.item()
            history_batch.append(loss.item())

            loss.backward()
            optimizer.step()

            learning_rate = optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch+1} / {options.n_epochs} Batch {batch_i+1} / {len(train_loader)} LR {learning_rate:0.7f} Loss {loss.item():0.7f}", end="\r")
        scheduler.step()
        print("")

    return history_epoch, history_batch


def test_model(model: LanguageClassifier, test_loader, loss_function, test_losses, options: Options):
    model.eval()
    test_acc = 0
    temp_losses = []

    for data, labels in test_loader:
        data = data.to(options.device)
        labels = labels.to(options.device)
        predictions = model(data)

        accuracy = (torch.max(predictions, dim=-1, keepdim=True)
                    [1].flatten() == labels).sum() / len(labels)
        test_acc += accuracy.item()

    test_acc /= len(test_loader)
    return f"Test accuracy {test_acc}"


def build_model(options: Options, train_loader, test_loader, save_model=True, checkpoint_dir=None):
    train_losses = []
    test_losses = []

    weights = torch.tensor([1., 1., 1., 1., 1., 1.])
    weights = weights.to(options.device)

    loss_function = nn.CrossEntropyLoss(
        weight=weights)

    model = LanguageClassifier(
        options) if checkpoint_dir is None else load_jit_model(options, checkpoint_dir)

    history_epoch, history_batch = train_model(
        model, train_loader, loss_function, options)

    if save_model:
        save_model_weights(model, options, f"model_state_dict.pt")

    print(test_model(model, test_loader, loss_function, test_losses, options))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_epoch, label="train")
    plt.title("Epoch loss history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_batch, label="train")
    plt.title("Batch loss history")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

    return model, train_losses, test_losses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    options = Options(use_all_languages=True)
    train_loader, test_loader = get_dataloaders(options)
    model, _, _ = build_model(options, train_loader, test_loader,
                              save_model=True)
