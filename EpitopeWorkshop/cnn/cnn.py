from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from EpitopeWorkshop.common import contract

KERNEL_SIZE = 3
IN_CHANNELS = 1
LAYER_1_CHANNELS = 6
OUT_CHANNELS = 10
PADDING = 1
CLASSIFICATION_OPTIONS_AMT = 1


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                IN_CHANNELS, out_channels=LAYER_1_CHANNELS,
                kernel_size=KERNEL_SIZE, padding=PADDING
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=LAYER_1_CHANNELS, out_channels=OUT_CHANNELS,
                kernel_size=KERNEL_SIZE, padding=PADDING),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(OUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * len(contract.FEATURES_ORDERED), 120),
            nn.ReLU(),
            nn.Linear(120, CLASSIFICATION_OPTIONS_AMT),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        probability = self.classifier(features)
        return probability

    def to_pth(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pth(cls, path: str) -> 'CNN':
        cnn = cls()
        cnn.load_state_dict(torch.load(path))
        return cnn
