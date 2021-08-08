import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=6, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=10, kernel_size=4),
            nn.ReLU(),


        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 1), # maybe one more Linear?
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        probability = self.classifier(features)
        return probability


net = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)