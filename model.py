from options import Options
import torch
import torch.nn as nn
import torch.nn.functional

class BinaryLanguageClassifier(nn.Module):
    def __init__(self, options: Options):
        super().__init__()

        # these layers are not correct, but just something as an example

        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # these layers are not correct, but just something as an example

        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

def build_model(options: Options, X_train, y_train, X_test, y_test):

    # train the model here, including defining loss function, optimizer, etc

    model = BinaryLanguageClassifier(options)
    
    return model
