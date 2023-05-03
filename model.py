from options import Options
import torch
import torch.nn as nn
import torch.optim as optim
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

def train_model(model, device, train_loader):
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Another optimiser - Adam
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
    crossentropy_loss = nn.CrossEntropyLoss() 
    num_epochs = 3
    model.train()
    for epoch in range(num_epochs):
        for data, labels in train_loader: 
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()           
            predictions = model(data)     
            loss = crossentropy_loss(predictions, labels) 
            loss.backward() 
            optimizer.step()
    
    return model

def test_model(model, device, test_loader):
    model.eval() 
    test_acc = 0 
    for data, labels in test_loader: 
        data = data.to(device)
        labels = labels.to(device)
        predictions = model(data)
        accuracy = (torch.max(predictions, dim=-1, keepdim=True)[1].flatten() == labels).sum() / len(labels)
        test_acc += accuracy.item()
    test_acc /= len(test_loader) 
    return f"Test accuracy {test_acc}"


def build_model(options: Options, train_loader, test_loader):
    # train the model here, including defining loss function, optimizer, etc
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BinaryLanguageClassifier(options)
    train_model(model, device, train_loader)
    test_model(model, device, test_loader)
        
    
    return model

