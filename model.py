from options import Options
import torch.nn as nn
import torch.nn.functional as F

    
class BinaryLanguageClassifier(nn.Module):
    def __init__(self, options: Options, n_input=40000, n_output=1, stride=16, n_channel=32):
        super().__init__()

        # some of these layers might be useful

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        #self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4) # reduces the dimensionallity 
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(n_channel, 2*n_channel, kernel_size=80, stride=stride)
        #self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4) # reduces the dimensionallity 
        self.relu2 = nn.ReLU()

        self.Linear_2 = nn.Linear(in_features=2*n_channel, out_features=n_output)
        
    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        return F.log_softmax(x, dim=2)

def build_model(options: Options, train_loader, test_loader):
    # train the model here, including defining loss function, optimizer, etc
    model = BinaryLanguageClassifier(options)
    
    return model

