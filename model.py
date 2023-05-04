from options import Options
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
    
class BinaryLanguageClassifier(nn.Module):
    def __init__(self, options: Options):
        super().__init__()

        self.mel_spectogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=options.sample_rate)
        # x = self.mel_spectogram_transform(x) # x is of shape (32 x 128 x 201)

        output_size = 1
        stride=16

        self.conv1 = nn.Conv1d(1, output_size, kernel_size=40, stride=stride)
        #self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4) # reduces the dimensionallity 
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(1, 1, kernel_size=120, stride=stride)
        #self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(5) # reduces the dimensionallity 
        self.relu2 = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        # x is of shape [batch_size, input_size] (32 x 40000)
        x = x.unsqueeze(1) # x is of shape (32 x 1 x 40000)
        x = self.conv1(x) # x is of shape (32 x 1 x 2498)
        x = self.pool1(x) # x is of shape (32 x 1 x 624)
        x = self.conv2(x) # x is of shape (32 x 1 x 32)
        x = self.pool2(x) # x is of shape (32 x 1 x 6)
        x = x.squeeze(1) # x is of shape (32 x 6)
        return F.log_softmax(x, dim=1)
    

def train_model(model, train_loader, options: Options):
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr) # Another optimiser - Adam
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
    crossentropy_loss = nn.CrossEntropyLoss() 
    
    model.train()
    for epoch in range(options.n_epochs):
        for data, labels in train_loader: 
            data = data.to(options.device)
            labels = labels.to(options.device)
            optimizer.zero_grad()           
            predictions = model(data)     
            loss = crossentropy_loss(predictions, labels) 
            loss.backward() 
            optimizer.step()
    
    return model

def test_model(model, test_loader, options: Options):
    model.eval() 
    test_acc = 0 

    for data, labels in test_loader: 
        data = data.to(options.device)
        labels = labels.to(options.device)
        predictions = model(data)
        accuracy = (torch.max(predictions, dim=-1, keepdim=True)[1].flatten() == labels).sum() / len(labels)
        test_acc += accuracy.item()
    test_acc /= len(test_loader) 
    return f"Test accuracy {test_acc}"


def build_model(options: Options, train_loader, test_loader):
    # train the model here, including defining loss function, optimizer, etc
    model = BinaryLanguageClassifier(options)
    train_model(model, train_loader, options)
    print(test_model(model, test_loader, options))
    return model

