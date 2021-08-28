import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd

KERNEL_SIZE = 3
OUT_CHANNELS = 10


class CNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=6, kernel_size=KERNEL_SIZE, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=1),
            nn.ReLU(),

        )
        self.classifier = nn.Sequential(
            nn.Linear(OUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * 26, 120),
            nn.ReLU(),
            nn.Linear(120, 1),
            nn.Sigmoid()
        )

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        probability = self.classifier(features)
        return probability
