import glob
import logging
import math
import os
import pickle
import random
from typing import Optional

import torch
import numpy as np
import pandas as pd
from EpitopeWorkshop.cnn.cnn import CNN
from EpitopeWorkshop.cnn.train import ModelTrainer
from EpitopeWorkshop.common import contract, plot
from EpitopeWorkshop.common.conf import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_IS_IN_EPITOPE_THRESHOLD, \
    DEFAULT_PRESERVE_FILES_IN_PROCESS
from EpitopeWorkshop.dataset.EpitopeDataset import EpitopeDataset


def load_dataset(df_path: str) -> EpitopeDataset:
    df = pd.read_pickle(df_path)
    calculated_features = df[contract.CALCULATED_FEATURES_COL_NAME]
    labels = df[contract.IS_IN_EPITOPE_COL_NAME]
    ds = EpitopeDataset(calculated_features, labels)
    return ds


def load_df_as_dl(path: str, batch_size: int, limit: Optional[int] = None):
    with open(path, 'rb') as fp:
        df = pickle.load(fp)  # type: pd.DataFrame
    if limit is not None:
        keep_proba = limit / len(df)
        msk = np.random.rand(len(df)) < keep_proba
        df = df[msk].reset_index(drop=True)
    ds = EpitopeDataset(
        df[contract.CALCULATED_FEATURES_COL_NAME],
        df[contract.IS_IN_EPITOPE_COL_NAME]
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                       shuffle=True, num_workers=0), df


class Train:
    def train(self, train_files_dir: str, validation_files_dir: str, test_files_dir: str, epochs: int = DEFAULT_EPOCHS,
              persist_cnn_path: Optional[str] = None, batch_size: int = DEFAULT_BATCH_SIZE,
              pos_weight: Optional[float] = None, preserve_files_in_process: bool = DEFAULT_PRESERVE_FILES_IN_PROCESS):
        cnn = CNN()
        trainer = ModelTrainer(cnn, pos_weight)

        train_files = glob.glob(os.path.join(train_files_dir, '*.df'))
        test_files = glob.glob(os.path.join(test_files_dir, '*.df'))
        for epoch in range(epochs):
            logging.info(f"running on all train data, epoch {epoch}")
            random.shuffle(train_files)
            for index, file in enumerate(train_files):
                logging.info(f"training file ({index + 1}/{len(train_files)}) {file}")
                random.shuffle(test_files)
                cur_test_files = test_files[:2]
                dl_train, _ = load_df_as_dl(file, batch_size)
                dls_dfs = [load_df_as_dl(test_file, batch_size) for test_file in cur_test_files]
                dls_test = [x[0] for x in dls_dfs]
                train_accuracy, train_loss, test_accuracies, test_losses = \
                    trainer.train(batch_size, dl_train, dls_test, epoch_amt=1)
                plot.plot_training_data(test_accuracies, test_losses, train_accuracy, train_loss)
                if epoch == epochs-1 and not preserve_files_in_process:
                    logging.info(f"removing file {file}")
                    os.remove(file)
                if persist_cnn_path is not None:
                    dir_name, basename = os.path.split(persist_cnn_path)
                    final_path = os.path.join(dir_name, f"{epoch}-{basename}")
                    logging.info(f"persisting cnn (for epoch {epoch}) to disk to {final_path}")
                    cnn.to_pth(final_path)

        logging.info("done training cnn")

    def test_trained_model(self, pth_path: str, test_files_dir: str, batch_size: int = DEFAULT_BATCH_SIZE,
                           threshold: float = DEFAULT_IS_IN_EPITOPE_THRESHOLD,
                           limit_test_file_freq: Optional[float] = None):
        total_records = 0
        total_success = 0
        total_positive_records = 0
        total_positive_success = 0
        test_files = glob.glob(os.path.join(test_files_dir, '*'))
        random.shuffle(test_files)
        if limit_test_file_freq is not None:
            last_index = math.floor(len(test_files) * limit_test_file_freq)
            test_files = test_files[:last_index]
        cnn = CNN.from_pth(pth_path)
        for index, file in enumerate(test_files):
            file_records = 0
            file_positive_records = 0
            file_success = 0
            file_positive_success = 0
            logging.info(f"testing file ({index + 1}/{len(test_files)}) {file}")
            dl_test, _ = load_df_as_dl(file, batch_size)
            dl_test_iter = iter(dl_test)
            for test_batch in dl_test_iter:
                test_X, test_y = test_batch[0], test_batch[1]
                test_pred_proba = torch.sigmoid(cnn(test_X))
                test_prediction = (test_pred_proba >= threshold).int().squeeze()
                file_success += torch.sum(test_prediction == test_y).float().item()
                file_positive_success += torch.sum(test_prediction + test_y == 2).float().item()
                file_positive_records += torch.sum(test_y == 1).float().item()
                file_records += len(test_X)
            logging.info(
                f"file records: {file_records}, positive records: {file_positive_records}, success: {file_success}. Succes rate: {file_success / file_records}. Sucess rate for positive labels: {file_positive_success / max(1, file_positive_records)}")
            total_records += file_records
            total_success += file_success
            total_positive_records += file_positive_records
            total_positive_success += file_positive_success
        logging.info(
            f"total record: {total_records}, success: {total_success}. Success rate: {total_success / total_records}. Sucess rate for positive labels: {total_positive_success / max(1, total_positive_records)}")
