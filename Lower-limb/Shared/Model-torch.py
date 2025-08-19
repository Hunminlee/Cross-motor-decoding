import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        # input_shape = (channels, height, width) in PyTorch
        in_channels = input_shape[0]

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # replaces GlobalAveragePooling2D

        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # Conv + ReLU
        x = self.pool(x)               # MaxPooling
        x = F.relu(self.conv2(x))      # Conv + ReLU
        x = self.global_avg_pool(x)    # Global Avg Pool
        x = x.view(x.size(0), -1)      # flatten (batch_size, 32)
        x = F.relu(self.fc1(x))        # Dense 64
        x = self.fc2(x)                # Dense num_classes
        return x