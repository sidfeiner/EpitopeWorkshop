import glob
import logging
import math
import os
import pickle
import random
from typing import Optional, List

import torch
import numpy as np
import pandas as pd
from EpitopeWorkshop.cnn.cnn import CNN
from EpitopeWorkshop.cnn.train import ModelTrainer
from EpitopeWorkshop.common import contract, plot
from EpitopeWorkshop.common.conf import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_IS_IN_EPITOPE_THRESHOLD, \
    DEFAULT_PRESERVE_FILES_IN_PROCESS, DEFAULT_WEIGHT_DECAY, PATH_TO_CNN_DIR, DEFAULT_BATCHES_UNTIL_TEST, \
    DEFAULT_BALANCING_METHOD, DEFAULT_NORMALIZE_SURFACE_ACCESSIBILITY, DEFAULT_NORMALIZE_VOLUME, \
    DEFAULT_NORMALIZE_HYDROPHOBICITY
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
                                       shuffle=True, num_workers=0, drop_last=True), df


def load_many_dfs_as_dls(files: List[str], batch_size: int = DEFAULT_BATCH_SIZE):
    dls_dfs = (load_df_as_dl(file, batch_size) for file in files)
    return (x[0] for x in dls_dfs)


class Train:
    """Train a CNN with all train and test files"""

    def train(self, train_files_dir: str, validation_files_dir: str, test_files_dir: str, epochs: int = DEFAULT_EPOCHS,
              persist_cnn_path: Optional[str] = None, batch_size: int = DEFAULT_BATCH_SIZE,
              pos_weight: Optional[float] = None, weight_decay: Optional[float] = DEFAULT_WEIGHT_DECAY,
              normalize_hydrophobicity: bool = DEFAULT_NORMALIZE_HYDROPHOBICITY,
              normalize_volume: bool = DEFAULT_NORMALIZE_VOLUME,
              normalize_surface_accessibility: bool = DEFAULT_NORMALIZE_SURFACE_ACCESSIBILITY,
              threshold=DEFAULT_IS_IN_EPITOPE_THRESHOLD,
              preserve_files_in_process: bool = DEFAULT_PRESERVE_FILES_IN_PROCESS):
        """
        :param train_files_dir: Directory with all train dataframes
        :param validation_files_dir: Directory with all validation dataframes
        :param test_files_dir: Directory with all test dataframes
        :param epochs: Amount of epochs for learning
        :param persist_cnn_path: Path to persist the cnn to when learning
        :param batch_size: Batch size for the Data Loader - Defaults to 10
        :param pos_weight: If given, will give a positive weight to the loss func
        :param weight_decay: regularization parameter, defaults to 0.01
        :param normalize_hydrophobicity: If true, hydrophobicity values will be normalized during pre-process in CNN
        :param normalize_volume: If true, amino acid volume values will be normalized during pre-process in CNN
        :param normalize_surface_accessibility: If true, amino acid SA values will be normalized during pre-process in CNN
        :param threshold: Threshold to determine if an amino acid is part of the epitope
        :param preserve_files_in_process: If False, will delete training file after it has been learned
        """
        cnn = CNN(normalize_hydrophobicity, normalize_volume, normalize_surface_accessibility)
        trainer = ModelTrainer(cnn, pos_weight, weight_decay)

        train_files = glob.glob(os.path.join(train_files_dir, '*.df'))
        test_files = glob.glob(os.path.join(test_files_dir, '*.df'))
        validation_files = glob.glob(os.path.join(validation_files_dir, '*.df'))
        all_epochs_train_accuracy, all_epochs_train_loss, all_epochs_validation_acccuracy, all_epochs_validation_loss, \
        all_epochs_test_accuracy, all_epochs_test_loss = [], [], [], [], [], []
        for epoch in range(epochs):
            logging.info(f"running on all train data, epoch {epoch + 1}/{epochs}")
            random.shuffle(train_files)
            total_train_acc_sum, total_train_loss_sum = 0, 0
            train_batches_amt = 0
            for index, file in enumerate(train_files):
                logging.info(f"training file ({index + 1}/{len(train_files)}) {file}")
                random.shuffle(test_files)
                dl_train, _ = load_df_as_dl(file, batch_size)
                train_batches_amt += len(dl_train)
                train_acc_sum, train_loss_sum = trainer.train(dl_train, epoch_amt=1)
                total_train_acc_sum += train_acc_sum
                total_train_loss_sum += train_loss_sum
                if epoch == epochs - 1 and not preserve_files_in_process:
                    logging.info(f"removing file {file}")
                    os.remove(file)
                if persist_cnn_path is not None:
                    dir_name, basename = os.path.split(persist_cnn_path)
                    final_path = os.path.join(dir_name, f"{epoch}-{basename}")
                    logging.info(f"persisting cnn (for epoch {epoch}) to disk to {final_path}")
                    cnn.to_pth(final_path)

            dls_test = lambda: load_many_dfs_as_dls(test_files, batch_size)
            dls_validation = lambda: load_many_dfs_as_dls(validation_files, batch_size)

            logging.debug(f"running cnn on test")
            dls_test_len, test_total_loss, test_total_acc = trainer.get_loss_and_accuracy(dls_test, threshold)
            logging.debug(f"running cnn on validation")
            dls_validation_len, validation_total_loss, validation_total_acc = trainer.get_loss_and_accuracy(
                dls_validation, threshold)

            train_accuracy = total_train_acc_sum / (train_batches_amt * batch_size)
            train_loss = total_train_loss_sum / train_batches_amt
            test_accuracy = test_total_acc / (dls_test_len * batch_size)
            test_loss = test_total_loss / dls_test_len
            validation_accuracy = validation_total_acc / (dls_validation_len * batch_size)
            validation_loss = validation_total_loss / dls_validation_len
            logging.debug(
                f"for epoch {epoch}: train acc: {train_accuracy}, train loss: {train_loss}, test acc: {test_accuracy}, test loss: {test_loss}, validation acc: {validation_accuracy}, validation loss: {validation_loss}")

            all_epochs_train_accuracy.append(train_accuracy)
            all_epochs_train_loss.append(train_loss)
            all_epochs_test_accuracy.append(test_accuracy)
            all_epochs_test_loss.append(test_loss)
            all_epochs_validation_acccuracy.append(validation_accuracy)
            all_epochs_validation_loss.append(validation_loss)

        plot.plot_training_data(
            all_epochs_train_accuracy, all_epochs_test_loss,
            all_epochs_validation_acccuracy, all_epochs_validation_loss,
            all_epochs_train_accuracy, all_epochs_train_loss,
            f"summarize all {epochs} epochs, weight decay {weight_decay}"
        )
        logging.info("done training cnn")

    def test_trained_model(self, cnn_name: str, test_files_dir: str, batch_size: int = DEFAULT_BATCH_SIZE,
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
        cnn = CNN.from_pth(os.path.join(PATH_TO_CNN_DIR, cnn_name))
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
