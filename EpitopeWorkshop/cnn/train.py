import logging

import torch
import time

from torch.utils import data
from EpitopeWorkshop.common.conf import DEFAULT_EPOCHS
from EpitopeWorkshop.cnn.cnn import CNN

TEST_BATCH_SIZE = 1000


def train_model(model: 'CNN', dl_train: data.Dataset, epoch_amt: int = DEFAULT_EPOCHS):
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
            loss = model.loss_func(outputs, labels)
            running_loss += loss
            loss.backward()
            model.optimizer.step()

            # print statistics
            # running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def train(model: 'CNN', batch_size: int, dl_train: data.Dataset, dl_test: data.Dataset, epoch_amt: int = DEFAULT_EPOCHS,
          max_batches=20):
    test_accuracies, test_losses = [], []
    train_accuracy, train_loss = [], []

    for epoch_idx in range(epoch_amt):
        logging.info(f"running epoch {epoch_idx + 1}/{epoch_amt}")
        total_loss, n_correct = 0, 0
        start_timestamp = time.time()

        for total_batch_idx, batch in enumerate(dl_train):
            X, y = batch[0], batch[1]
            print(f"x: {X}")
            logging.info(f"y: {y[0]}")

            # Forward pass
            y_pred_log_proba = model(X)
            logging.info(f"y_pred_log_proba: {y_pred_log_proba[0]}")
            # Backward pass
            model.optimizer.zero_grad()
            loss = model.loss_func(y_pred_log_proba, y)
            loss.backward()

            # Weight updates
            model.optimizer.step()

            # Calculate accuracy
            total_loss += loss.item()
            y_pred = torch.argmax(y_pred_log_proba, dim=1)
            n_correct += torch.sum(y_pred == y).float().item()
            if (total_batch_idx + 1) % TEST_BATCH_SIZE == 0:
                logging.debug(f"comparing with all the test data at batch {total_batch_idx + 1}")
                dl_test_iter = iter(dl_test)
                test_total_acc, test_total_loss = 0, 0
                for test_batch in dl_test_iter:
                    test_X, test_y = test_batch[0], test_batch[1]
                    test_pred_log_proba = model(test_X)
                    test_predication = torch.argmax(test_pred_log_proba, dim=1)
                    loss = model.loss_func(test_pred_log_proba, test_y)
                    test_total_loss += loss.item()
                    test_total_acc += torch.sum(test_predication == test_y).float().item()
                test_accuracies.append(test_total_acc / (len(dl_test) * batch_size))
                test_losses.append(test_total_loss / len(dl_test))
                train_accuracy.append(n_correct / (TEST_BATCH_SIZE * batch_size))
                train_loss.append(total_loss / TEST_BATCH_SIZE)
                n_correct, total_loss = 0, 0
        print(
            f"Epoch #{epoch_idx}, loss={total_loss / (max_batches):.3f}, accuracy={n_correct / (max_batches * batch_size):.3f},elapsed={time.time() - start_timestamp:.1f} sec")

    return train_accuracy, train_loss, test_accuracies, test_losses