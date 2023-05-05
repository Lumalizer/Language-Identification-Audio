from options import Options
import torch
from torchaudio.transforms import MelSpectrogram
import torch.nn as nn
import os


class BinaryLanguageClassifier(nn.Module):
    def __init__(self, options: Options):
        super().__init__()
        self.options = options

        self.mel_spectogram_transform = MelSpectrogram(
            sample_rate=options.sample_rate)

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(1, 1, kernel_size=(4, 4))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(1, 1, kernel_size=(3, 3))
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.lin1 = nn.Linear(1440, 200)
        self.lin2 = nn.Linear(200, 2)

        self.to(options.device)

        print(f"Model initialized on {options.device}")

    def forward(self, x: torch.Tensor, debug=False):
        # x is of shape [batch_size, input_size] (32 x 40000)
        x = self.mel_spectogram_transform(x)  # x is of shape (32 x 128 x 201)

        x = x.unsqueeze(1)  # x is of shape (32 x 1 x 128 x 201)
        debug and print(x.shape)

        x = self.conv1(x)  # x is of shape (32 x 1 x 125 x 198)
        debug and print(x.shape)
        x = self.maxpool1(x)  # x is of shape (32 x 1 x 62 x 99)
        debug and print(x.shape)
        x = self.relu1(x)

        x = self.conv2(x)  # x is of shape (32 x 1 x 59 x 96)
        debug and print(x.shape)
        x = self.maxpool2(x)  # x is of shape (32 x 1 x 29 x 48)
        debug and print(x.shape)
        x = self.relu1(x)

        # disabled layergroup 3 for now

        x = x.flatten(2)  # x is of shape (32, 1, 1392)
        debug and print(x.shape)

        x = self.lin1(x)  # x is of shape (32, 1, 200)
        x = self.relu1(x)
        debug and print(x.shape)

        x = self.lin2(x)  # x is of shape (32, 1, 2)
        x = self.relu1(x)
        debug and print(x.shape)

        x = x.squeeze(1)  # x is of shape (32, 2)
        debug and print(x.shape)

        return x  # don't use softmax, because we use crossentropy loss which directly takes logits


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
    crossentropy_loss = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(options.n_epochs):
        for batch_i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(options.device)
            labels = labels.to(options.device)
            predictions = model(data)
            loss = crossentropy_loss(predictions, labels)
            # sometimes predictions are just around 0, which leads to random accuracy
            # could be due to vanishing gradients or too small numbers somehow ?
            # print(predictions)
            # print(loss)
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
    # train the model here, including defining loss function, optimizer, etc
    model = BinaryLanguageClassifier(options)
    train_model(model, train_loader, options)
    print(test_model(model, test_loader, options))
    return model
