import torch.nn as nn
import torch
import torch.nn.functional as F

class BobNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=96,
            kernel_size=7,
            stride=4,
            padding=1
        )
        self.norm = nn.LocalResponseNorm(
            size=5,
            alpha=0.0001,
            beta=0.75
        )
        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2
        ) 
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.fcn1 = nn.Linear(
            in_features =18816,
            out_features=512
        )
        self.fcn2 = nn.Linear(
            in_features=512,
            out_features=256
        )
        self.dropout = nn.Dropout(0.5)
        self.fcn3 = nn.Linear(
            in_features=256,
            out_features=2  # Num classes
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.norm(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #x = self.norm(x)

        x = x.view(-1, 18816) # 384 * 7 * 7
        x = self.fcn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fcn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fcn3(x)
        x = self.softmax(x)

        return x
    
