import pickle

import torch.nn as nn
import torch.optim as optim

from EpitopeWorkshop.common import contract

KERNEL_SIZE = 3
IN_CHANNELS = 1
LAYER_1_CHANNELS = 6
OUT_CHANNELS = 10
PADDING = 1
CLASSIFICATION_OPTIONS_AMT = 2


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
            nn.Sigmoid()
        )

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        probability = self.classifier(features)
        return probability

    def to_pickle_file(self, path: str):
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)

    @classmethod
    def from_pickle_file(cls, path: str) -> 'CNN':
        with open(path, 'rb') as fp:
            cnn = pickle.load(fp)
        return cnn
