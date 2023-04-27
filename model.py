from options import Options
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLanguageClassifier(nn.Module):
    def __init__(self, options: Options, n_input=1, n_output=1, stride=16, n_channel=32):
        super().__init__()

        # some of these layers might be useful

        # self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        # self.bn1 = nn.BatchNorm1d(n_channel)
        # self.pool1 = nn.MaxPool1d(4)
        # self.fc1 = nn.Linear(2 * n_channel, n_output)


    def forward(self, x):

        # won't work yet

        # x = self.conv1(x)
        # x = F.relu(self.bn1(x))
        # x = self.pool1(x)
        # x = F.avg_pool1d(x, x.shape[-1])
        # x = x.permute(0, 2, 1)
        # x = self.fc1(x)
        # return F.log_softmax(x, dim=2)

        return

def build_model(options: Options, X_train, y_train, X_test, y_test):
    # train the model here, including defining loss function, optimizer, etc
    model = BinaryLanguageClassifier(options)
    
    return model

