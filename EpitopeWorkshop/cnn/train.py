import logging
from typing import Optional, List, Callable

import torch

from torch import optim, nn
from torch.utils import data
from EpitopeWorkshop.common.conf import DEFAULT_EPOCHS, DEFAULT_IS_IN_EPITOPE_THRESHOLD, DEFAULT_BATCHES_UNTIL_TEST
from EpitopeWorkshop.cnn.cnn import CNN


class ModelTrainer:
    def __init__(self, model: CNN, pos_weight: Optional[float] = None, weight_decay: Optional[float] = 1e-2):
        self.model = model
        self.loss_func = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight)
        ) if pos_weight is not None else torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(
            model.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay
        ) if weight_decay is not None else optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def train_model(self, dl_train: data.Dataset, epoch_amt: int = DEFAULT_EPOCHS):
        for epoch in range(epoch_amt):  # loop over the dataset multiple times
            logging.info(f"running for epoch {epoch + 1}/{epoch_amt}")
            running_loss = 0.0
            for i, data in enumerate(dl_train):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels.unsqueeze(1).float())
                running_loss += loss
                loss.backward()
                self.optimizer.step()

                # print statistics
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    logging.debug(
                        '[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000)
                    )
                    running_loss = 0.0

    def get_loss_and_accuracy(self, dls: Callable[[], List[data.Dataset]], threshold: float):
        total_acc, total_loss = 0, 0
        dls_len = 0
        for dl in dls():
            dl_test_iter = iter(dl)
            dls_len += len(dl)
            for test_batch in dl_test_iter:
                test_X, test_y = test_batch[0], test_batch[1]
                test_pred_log_proba = self.model(test_X)
                test_prediction = test_pred_log_proba >= threshold
                loss = self.loss_func(test_pred_log_proba, test_y.unsqueeze(1).float())
                total_loss += loss.item()
                total_acc += torch.sum(test_prediction == test_y.unsqueeze(1).float()).float().item()
        return dls_len, total_loss, total_acc

    def train(self, dl_train: data.Dataset, epoch_amt: int = DEFAULT_EPOCHS,
              threshold=DEFAULT_IS_IN_EPITOPE_THRESHOLD):

        total_accuracy, total_loss = 0, 0

        for epoch_idx in range(epoch_amt):
            logging.info(f"running epoch {epoch_idx + 1}/{epoch_amt}")
            for total_batch_idx, batch in enumerate(dl_train):
                if total_batch_idx % 2000 == 0:
                    logging.debug(f"running batch {total_batch_idx + 1}/{len(dl_train)}")
                X, y = batch[0], batch[1]

                # Forward pass
                y_pred_log_proba = self.model(X)

                # Backward pass
                self.optimizer.zero_grad()
                loss = self.loss_func(y_pred_log_proba, y.unsqueeze(1).float())
                loss.backward()

                # Weight updates
                self.optimizer.step()

                # Calculate accuracy
                total_loss += loss.item()
                y_pred = y_pred_log_proba >= threshold
                total_accuracy += torch.sum(y_pred == y.unsqueeze(1).float()).float().item()

        return total_accuracy, total_loss
