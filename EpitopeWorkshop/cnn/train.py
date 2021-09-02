import torch
import time
import logging

from torch.utils import data
from EpitopeWorkshop.common.conf import DEFAULT_EPOCHS
from EpitopeWorkshop.cnn.cnn import CNN

log_format = "%(asctime)s : %(threadName)s: %(levelname)s : %(name)s : %(module)s : %(message)s"
logging.basicConfig(format=log_format, level=logging.DEBUG)

TEST_BATCH_SIZE = 20
BATCH_SIZE = 4


def train_model(model: 'CNN', dl_train: data.Dataset, dl_test: data.Dataset, epoch_amt: int = DEFAULT_EPOCHS):
    test_accuracy, test_loss = [], []
    train_accuracy, train_loss = [], []

    for epoch in range(epoch_amt):  # loop over the dataset multiple times
        logging.debug(f"running for epoch {epoch + 1}")
        total_loss, n_correct = 0.0, 0.0
        start_timestamp = time.time()
        for i, data in enumerate(dl_train):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            model.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = model.loss_func(outputs, labels)
            loss.backward()
            model.optimizer.step()

            # Calculate accuracy
            total_loss += loss.item()
            y_pred = torch.argmax(outputs, dim=1)
            n_correct += torch.sum(y_pred == labels).float().item()
            if (i + 1) % TEST_BATCH_SIZE == 0:
                dl_test_iter = iter(dl_test)
                test_total_acc, test_total_loss = 0, 0
                for test_batch in dl_test_iter:
                    test_X, test_y = test_batch.text, test_batch.label
                    test_pred_log_proba = model(test_X)
                    test_predication = torch.argmax(test_pred_log_proba, dim=1)
                    loss = model.criterion(test_pred_log_proba, test_y)
                    test_total_loss += loss.item()
                    test_total_acc += torch.sum(test_predication == test_y).float().item()
                test_accuracy.append(test_total_acc / (len(dl_test) * BATCH_SIZE))
                test_loss.append(test_total_loss / len(dl_test))
                train_accuracy.append(n_correct / (TEST_BATCH_SIZE * BATCH_SIZE))
                train_loss.append(total_loss / TEST_BATCH_SIZE)
        logging.debug(
            f"Epoch #{epoch}, loss={total_loss / (TEST_BATCH_SIZE):.3f}, accuracy={n_correct / (TEST_BATCH_SIZE * BATCH_SIZE):.3f},elapsed={time.time() - start_timestamp:.1f} sec")

    return train_accuracy, train_loss, test_accuracy, test_loss

