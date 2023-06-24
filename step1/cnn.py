import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # conv1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # conv2
        self.conv3 = nn.Conv2d(64, 128, 3, 1)  # conv3
        self.dropout1 = nn.Dropout(0.25)  # dropout1
        self.dropout2 = nn.Dropout(0.5)  # dropout2
        self.adaptive_pool = nn.AdaptiveAvgPool2d((15,15))  # pooling layer
        self.fc1 = nn.Linear(128*15*15, 128)  #fc1
        self.fc2 = nn.Linear(128, 10)  #fc2 output 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)  # flattening for linear layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
