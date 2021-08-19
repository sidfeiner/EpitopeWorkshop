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
            nn.Linear(120, 1),
            nn.Sigmoid()
        )

@classmethod
def forward(self, x):
    features = self.feature_extractor(x)
    features = features.view(features.size(0), -1)
    probability = self.classifier(features)
    return probability


net = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

@classmethod
def trainingLoop(model, optimizer):
    train_size = len(dl_train)
    num_epochs = 100
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch_idx in range(num_epochs):
        epoch_train_losses = 0
        epoch_train_accs = 0
        epoch_test_losses = 0
        epoch_test_accs = 0

        for batch_idx, (X, y) in enumerate(dl_train):
            # Forward pass
            y_pred = model(X)

            # Compute loss
            loss = loss_fn(y_pred, y)
            epoch_train_losses += loss.item()
            epoch_train_accs += acc_batch(y_pred, y)

            # Backward pass
            optimizer.zero_grad()  # Zero gradients of all parameters
            loss.backward()  # Run backprop algorithms to calculate gradients

            # Optimization step
            optimizer.step()  # Use gradients to update model parameters
        # print(f'Epoch #{epoch_idx+1}: Avg. loss={total_loss/len(dl_train)}')
        for i, (x_test, y_test) in enumerate(dl_test):
            pred_test = model(x_test)
            epoch_test_losses += loss_fn(pred_test, y_test).item()
            epoch_test_accs += acc_batch(pred_test, y_test)

        if (epoch_idx + 1) % 20 == 0:
            # print(f'20 Batches #{epoch_idx//2 + 1}: Avg. loss={train_batch_20_losses/20}')
            # print(f'20 Batches #{epoch_idx//2 + 1}: Avg. accuracy={train_batch_20_accs/20}')
            train_losses.append(epoch_train_losses / train_size)
            train_accs.append(epoch_train_accs / train_size)
            test_losses.append(epoch_test_losses / test_size)
            test_accs.append(epoch_test_accs / test_size)
    return train_losses, train_accs, test_losses, test_accs