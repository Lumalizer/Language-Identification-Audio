import torch
import torch.nn as nn
from options import Options
from torchaudio.transforms import MFCC
import logging
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass
import torch
import numpy as np
from options import Options
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Options:
    num_languages: int = 6
    normalize: bool = True
    batch_size: int = 64
    input_size: int = 40000
    sample_rate: int = 8000
    n_epochs: int = 30
    lr: float = 0.0001
    model_path: str = "model"
    device: str = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    lstm_hidden_size: int = 312


def get_dataloaders(options: Options):
    X_train, y_train = np.load("dataset/inputs_train_fp16.npy"), np.load(
        "dataset/targets_train_int8.npy")
    X_test, y_test = np.load("dataset/inputs_test_fp16.npy"), np.load(
        "dataset/targets_test_int8.npy")

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    train_dataset = TensorDataset(torch.tensor(
        X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(
        X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(
        train_dataset, batch_size=options.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=options.batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


class LanguageClassifier(nn.Module):
    def __init__(self, options: Options):
        super(LanguageClassifier, self).__init__()
        logging.info("Initializing model...")

        self.options = options

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

        self.lstm = nn.LSTM(input_size=1152, hidden_size=options.lstm_hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(options.lstm_hidden_size *
                            2, options.num_languages)

        self.to(options.device)
        logging.info(f"Model initialized on {options.device}")

    def forward(self, x: torch.Tensor):
        # torch no_grad is used since we don't want to save the gradient for the pre-processing steps
        with torch.no_grad():
            x = F.normalize(x, p=2.0, dim=1)  # L2 normalization
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

        x = self.dropout(x)
        x = self.fc(x)
        return x


def save_jit_model(model: nn.Module, options: Options, name="JIT_model_state_dict.pt"):
    if not os.path.exists(options.model_path):
        os.makedirs(options.model_path)
    path = os.path.join(options.model_path, name)
    logging.info(f"Saving JIT model to {path}")
    torch.jit.save(torch.jit.script(model), path)
    logging.info(f"Model saved at {path}")


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


def test_model(model: LanguageClassifier, test_loader, options: Options):
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
    print(f"Test accuracy: {test_acc:0.7f}")


def build_model(options: Options, train_loader, test_loader, save_model=True, checkpoint_dir=None, plot_history=True):

    weights = torch.tensor([1., 1., 1., 1., 1., 1.])
    weights = weights.to(options.device)

    loss_function = nn.CrossEntropyLoss(
        weight=weights)

    model = LanguageClassifier(
        options) if checkpoint_dir is None else load_jit_model(options, checkpoint_dir)

    history_epoch, history_batch = train_model(
        model, train_loader, loss_function, options)

    if save_model:
        save_jit_model(model, options)

    test_model(model, test_loader, options)

    if plot_history:
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

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    options = Options()
    train_loader, test_loader = get_dataloaders(options)
    model = build_model(options, train_loader, test_loader, save_model=True)
