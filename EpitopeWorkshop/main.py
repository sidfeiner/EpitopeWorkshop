import glob
import logging
import os

import fire
import pandas as pd
import matplotlib.pyplot as plt

from EpitopeWorkshop.cnn.train import train_model, train
from EpitopeWorkshop.common import contract
from EpitopeWorkshop.dataset.EpitopeDataset import EpitopeDataset
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.cnn.cnn import CNN
from EpitopeWorkshop.scripts.calculate_features import FileFeatureCalculator
from EpitopeWorkshop.scripts.over_balance import OverBalancer

log_format = "%(asctime)s : %(threadName)s: %(levelname)s : %(name)s : %(module)s : %(message)s"
logging.basicConfig(format=log_format, level=logging.DEBUG)


def plot_graph(test_accs, test_losses, train_accs, train_losses):
    fig = plt.figure()
    plt.plot(train_accs)
    plt.plot(train_losses)
    plt.plot(test_accs)
    plt.plot(test_losses)
    plt.xlabel('# Batch')
    plt.legend(['Train accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss'], fontsize=13)
    fig.tight_layout()
    plt.show()


class Epitopes(OverBalancer, FileFeatureCalculator):

    def test(self, balanced_data_dir: str):
        files = glob.glob(os.path.join(balanced_data_dir, '*balanced*.fasta'))
        for file in files:
            logging.info(f"testing file {file}")
            logging.info("reading file")
            df = pd.read_pickle(file)
            calculated_features = df[contract.CALCULATED_FEATURES_COL_NAME]
            labels = df[contract.IS_IN_EPITOPE_COL_NAME]
            ds = EpitopeDataset(calculated_features, labels)

            logging.info("splitting to train, valid, test")
            dl_train, dl_valid, dl_test = ds.iters(batch_size=DEFAULT_BATCH_SIZE)

            cnn = CNN()
            logging.info("learning")
            train_accuracy, train_loss, test_accuracies, test_losses = train(cnn, DEFAULT_BATCH_SIZE, dl_train, dl_test)
            plot_graph(test_accuracies, test_losses, train_accuracy, train_loss)



if __name__ == '__main__':
    fire.Fire(Epitopes)
