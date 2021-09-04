import glob
import logging
import os
import pickle
import random
from typing import Optional

import fire
import pandas as pd
import torch

from EpitopeWorkshop.cnn.train import train_model, train
from EpitopeWorkshop.common import contract, plot
from EpitopeWorkshop.dataset.EpitopeDataset import EpitopeDataset
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.cnn.cnn import CNN
from EpitopeWorkshop.scripts.calculate_features import FileFeatureCalculator
from EpitopeWorkshop.scripts.over_balance import OverBalancer
from EpitopeWorkshop.scripts.df_to_csv import DFToCSV
from EpitopeWorkshop.scripts.split_data import SplitData

log_format = "%(asctime)s : %(threadName)s: %(levelname)s : %(name)s : %(module)s : %(message)s"
logging.basicConfig(format=log_format, level=logging.DEBUG)


def load_dataset(df_path: str) -> EpitopeDataset:
    df = pd.read_pickle(df_path)
    calculated_features = df[contract.CALCULATED_FEATURES_COL_NAME]
    labels = df[contract.IS_IN_EPITOPE_COL_NAME]
    ds = EpitopeDataset(calculated_features, labels)
    return ds


def load_df_as_dl(path: str, batch_size: int):
    with open(path, 'rb') as fp:
        df_train = pickle.load(fp)
    ds_train = EpitopeDataset(
        df_train[contract.CALCULATED_FEATURES_COL_NAME],
        df_train[contract.IS_IN_EPITOPE_COL_NAME]
    )
    return torch.utils.data.DataLoader(ds_train, batch_size=batch_size,
                                       shuffle=True, num_workers=0)


class Epitopes(OverBalancer, FileFeatureCalculator, DFToCSV, SplitData):

    def test(self, balanced_data_dir: str, pos_weight: Optional[float] = None):
        files = glob.glob(os.path.join(balanced_data_dir, '*balanced*.fasta'))
        for file in files:
            logging.info(f"testing file {file}")
            logging.info("reading file")
            ds = load_dataset(file)

            logging.info("splitting to train, valid, test")
            dl_train, dl_valid, dl_test = ds.iters(batch_size=DEFAULT_BATCH_SIZE)

            cnn = CNN(pos_weight)
            logging.info("learning")
            train_accuracy, train_loss, test_accuracies, test_losses = train(cnn, DEFAULT_BATCH_SIZE, dl_train, dl_test)
            plot.plot_training_data(test_accuracies, test_losses, train_accuracy, train_loss)

    def train(self, train_files_dir: str, validation_files_dir: str, test_files_dir: str, epochs: int = DEFAULT_EPOCHS,
              persist_cnn_path: Optional[str] = None, batch_size: int = DEFAULT_BATCH_SIZE,
              pos_weight: Optional[float] = None):
        cnn = CNN(pos_weight)

        train_files = glob.glob(os.path.join(train_files_dir, '*.df'))[:1]
        for epoch in range(epochs):
            logging.info(f"running on all train data, epoch {epoch}")
            random.shuffle(train_files)
            for file in train_files:
                logging.info(f"training file {file}")
                dl_train = load_df_as_dl(file, batch_size)
                train_model(cnn, dl_train, epoch_amt=1)
        logging.info("done training cnn")
        if persist_cnn_path is not None:
            logging.info(f"persisting cnn to disk to {persist_cnn_path}")
            cnn.to_pth(persist_cnn_path)

    def test_trained_model(self, pth_path: str, test_files_dir: str, batch_size: int = DEFAULT_BATCH_SIZE,
                           pos_weight: Optional[float] = None):
        total_records = 0
        total_success = 0
        test_files = glob.glob(os.path.join(test_files_dir, '*'))

        cnn = CNN.from_pth(pth_path, pos_weight)
        for file in test_files:
            file_records = 0
            file_success = 0
            logging.info(f"testing file {file}")
            dl_test = load_df_as_dl(file, batch_size)
            dl_test_iter = iter(dl_test)
            for test_batch in dl_test_iter:
                test_X, test_y = test_batch[0], test_batch[1]
                test_pred_log_proba = cnn(test_X)
                test_predication = torch.argmax(test_pred_log_proba, dim=1)
                file_success += torch.sum(test_predication == test_y).float().item()
                file_records += len(test_X)
            logging.info(
                f"file records: {file_records}, success: {file_success}. Succes rate: {file_success / file_records}")
            total_records += file_records
            total_success += file_success
        logging.info(
            f"total record: {total_records}, success: {total_success}. Success rate: {total_success / total_records}")


if __name__ == '__main__':
    fire.Fire(Epitopes)
