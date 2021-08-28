from torch.utils import data
from EpitopeWorkshop.common.contract import *
from EpitopeWorkshop.common.conf import DEFAULT_EPOCHS
from EpitopeWorkshop.cnn.cnn import CNN


def train_model(model: 'CNN', dl_train: data.Dataset, epoch_amt: int = DEFAULT_EPOCHS):
    for epoch in range(epoch_amt):  # loop over the dataset multiple times
        print(f"running for epoch {epoch + 1}")
        running_loss = 0.0
        for i, data in enumerate(dl_train):
            print(f"start with batch {i}")
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            model.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # loss = model.criterion(outputs, labels)
            # loss.backward()
            model.optimizer.step()

            # print statistics
            # running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')