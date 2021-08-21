import torch
import torchvision
import torch.utils.data
import torchvision.transforms
from torch.utils import data
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from EpitopeWorkshop.common.contract import *
from EpitopeWorkshop.common.conf import DEFAULT_EPOCHS


class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=6, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=10, kernel_size=4),
            nn.ReLU(),

        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 1),
            nn.Sigmoid()
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        probability = self.classifier(features)
        return probability

    @staticmethod
    def training_loop(model: 'CNN', dl_train: data.Dataset, epoch_amt: int = DEFAULT_EPOCHS):
        for epoch in range(epoch_amt):  # loop over the dataset multiple times
            print(f"running for epoch {epoch + 1}")
            running_loss = 0.0
            for i, data in enumerate(dl_train):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                model.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = model.criterion(outputs, labels)
                loss.backward()
                model.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
