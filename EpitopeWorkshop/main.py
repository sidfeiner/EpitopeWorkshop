import glob
import logging
import os
import pickle
import random
from typing import Optional

import fire
import pandas as pd
import matplotlib.pyplot as plt

from EpitopeWorkshop.cnn.train import train_model, train
from EpitopeWorkshop.common import contract, utils
from EpitopeWorkshop.dataset.EpitopeDataset import EpitopeDataset
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.cnn.cnn import CNN
from EpitopeWorkshop.scripts.calculate_features import FileFeatureCalculator
from EpitopeWorkshop.scripts.over_balance import OverBalancer

log_format = "%(asctime)s : %(threadName)s: %(levelname)s : %(name)s : %(module)s : %(message)s"
logging.basicConfig(format=log_format, level=logging.DEBUG)


def load_dataset(df_path: str) -> EpitopeDataset:
    df = pd.read_pickle(df_path)
    calculated_features = df[contract.CALCULATED_FEATURES_COL_NAME]
    labels = df[contract.IS_IN_EPITOPE_COL_NAME]
    ds = EpitopeDataset(calculated_features, labels)
    return ds


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

    with open('C:\\Users\\User\\Downloads\\balanced.csv', 'rb')as file:
        logging.info(f"testing file {file}")
        logging.info("reading file")
        df = pd.read_csv(file)
        calculated_features = df[contract.CALCULATED_FEATURES_COL_NAME]
        labels = df[contract.IS_IN_EPITOPE_COL_NAME]
        ds = EpitopeDataset(calculated_features, labels)

        logging.info("splitting to train, valid, test")
        dl_train, dl_valid, dl_test = ds.iters(batch_size=DEFAULT_BATCH_SIZE)

        cnn = CNN()
        logging.info("learning")
        train_accuracy, train_loss, test_accuracies, test_losses = train(cnn, DEFAULT_BATCH_SIZE, dl_train, dl_test)
        plot_graph(test_accuracies, test_losses, train_accuracy, train_loss)

    def test(self, balanced_data_dir: str):
        files = glob.glob(os.path.join(balanced_data_dir, '*balanced*.fasta'))
        for file in files:
            logging.info(f"testing file {file}")
            logging.info("reading file")
            ds = load_dataset(file)

            logging.info("splitting to train, valid, test")
            dl_train, dl_valid, dl_test = ds.iters(batch_size=DEFAULT_BATCH_SIZE)

            cnn = CNN()
            logging.info("learning")
            train_accuracy, train_loss, test_accuracies, test_losses = train(cnn, DEFAULT_BATCH_SIZE, dl_train, dl_test)
            plot_graph(test_accuracies, test_losses, train_accuracy, train_loss)

    def train(self, balanced_data_dir: str, create_files: bool = False, epochs: int = DEFAULT_EPOCHS,
              persist_cnn_path: Optional[str] = None):
        files = glob.glob(os.path.join(balanced_data_dir, '*balanced*.fasta'))
        cnn = CNN()
        train_files_dir = os.path.join(balanced_data_dir, 'train-files')
        validation_files_dir = os.path.join(balanced_data_dir, 'validation-files')
        test_files_dir = os.path.join(balanced_data_dir, 'test-files')
        os.makedirs(train_files_dir, exist_ok=True)
        os.makedirs(validation_files_dir, exist_ok=True)
        os.makedirs(test_files_dir, exist_ok=True)
        if create_files:
            logging.info("creating train, valid, test directories and files")
            for file in files:
                logging.info(f"splitting data for file {file}")
                file_part_index = utils.parse_index_from_partial_data_file(file)
                ds = load_dataset(file)
                logging.info("splitting to train, valid, test")

                dl_train, dl_validation, dl_test = ds.iters(batch_size=DEFAULT_BATCH_SIZE)
                dl_train_path = os.path.join(train_files_dir, f'train-{file_part_index}.dl')
                dl_validation_path = os.path.join(validation_files_dir, f'validation-{file_part_index}.dl')
                dl_test_path = os.path.join(test_files_dir, f'train-{file_part_index}.dl')

                with open(dl_train_path, 'wb') as fp:
                    pickle.dump(dl_train, fp)
                with open(dl_validation_path, 'wb') as fp:
                    pickle.dump(dl_validation, fp)
                with open(dl_test_path, 'wb') as fp:
                    pickle.dump(dl_test, fp)

        train_files = glob.glob(os.path.join(train_files_dir, '*.dl'))
        for epoch in range(epochs):
            logging.info(f"running on all train data, epoch {epoch}")
            random.shuffle(train_files)
            for file in train_files:
                logging.info(f"training file {file}")
                with open(file, 'rb') as fp:
                    dl_train = pickle.load(fp)
                train_model(cnn, dl_train, epoch_amt=1)
        logging.info("done training cnn")
        if persist_cnn_path is not None:
            logging.info(f"persisting cnn to disk to {persist_cnn_path}")
            cnn.to_pickle_file(persist_cnn_path)

        # total_records = 0
        # total_success = 0
        # test_files = glob.glob(os.path.join(test_files_dir, '*.dl'))
        #
        # for file in test_files:
        #     with open(file, 'rb') as fp:
        #         dl_test = pickle.load(fp)
        #     dl_test_iter = iter(dl_test)
        #     for test_batch in dl_test_iter:
        #         test_X, test_y = test_batch[0], test_batch[1]
        #         test_pred_log_proba = cnn(test_X)
        #         test_predication = torch.argmax(test_pred_log_proba, dim=1)
        #         loss = cnn.loss_func(test_pred_log_proba, test_y)
        #         test_total_loss += loss.item()
        #         test_total_acc += torch.sum(test_predication == test_y).float().item()


if __name__ == '__main__':
    fire.Fire(Epitopes)
