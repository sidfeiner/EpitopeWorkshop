import glob
import logging
import os
import random
from typing import List
import pandas as pd

import torch

from EpitopeWorkshop.common import contract, conf
from EpitopeWorkshop.common.conf import DEFAULT_PRESERVE_FILES_IN_PROCESS


class FileStats:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.df_records = 0
        self.handled_records = 0
        self.df_errors = 0

    def __str__(self):
        return f"file: {self.file_name}, records: {self.df_records}, errors: {self.df_errors}"


class AllStats:
    def __init__(self):
        self.file_stats = []  # type: List[FileStats]
        self.created_train_dfs = 0
        self.created_validation_dfs = 0
        self.created_test_dfs = 0

    def append_stats(self, stats: FileStats):
        self.file_stats.append(stats)


class ShuffleData:
    def data_to_df_disk(self, path: str, tensors, labels):
        df = pd.DataFrame(
            {
                contract.CALCULATED_FEATURES_COL_NAME: tensors,
                contract.IS_IN_EPITOPE_COL_NAME: labels
            }
        )
        logging.info(f"persisting file to {path}, with {len(tensors)}  items")
        df.to_pickle(path)
        return df

    def shuffle_data_dir(self, balanced_files_dir: str, max_records_per_df: int = 350000,
                         preserve_files_in_process: bool = DEFAULT_PRESERVE_FILES_IN_PROCESS):
        files = glob.glob(os.path.join(balanced_files_dir, '*.fasta'))
        train_files = [[] for _ in range(25)]  # type: List[List[torch.FloatTensor, int]]
        validation_files = [[] for _ in range(5)]  # type: List[List[torch.FloatTensor, int]]
        test_files = [[] for _ in range(10)]  # type: List[List[torch.FloatTensor, int]]

        train_files_dir = os.path.join(balanced_files_dir, 'train-files')
        validation_files_dir = os.path.join(balanced_files_dir, 'validation-files')
        test_files_dir = os.path.join(balanced_files_dir, 'test-files')
        os.makedirs(train_files_dir, exist_ok=True)
        os.makedirs(validation_files_dir, exist_ok=True)
        os.makedirs(test_files_dir, exist_ok=True)

        all_stats = AllStats()

        def persist_file_data(lst: list, index: int):
            if lst == train_files:
                final_dir = train_files_dir
                file_idx = all_stats.created_train_dfs
                all_stats.created_train_dfs += 1
            elif lst == validation_files:
                final_dir = validation_files_dir
                file_idx = all_stats.created_validation_dfs
                all_stats.created_validation_dfs += 1
            else:
                final_dir = test_files_dir
                file_idx = all_stats.created_test_dfs
                all_stats.created_test_dfs += 1

            batch_tensors = []
            batch_labels = []
            for tensor, label in lst[index]:
                batch_tensors.append(tensor)
                batch_labels.append(label)
            final_df_path = os.path.join(final_dir,
                                         f"iedb_linear_epitopes_{file_idx}_shuffled.df")
            self.data_to_df_disk(final_df_path, batch_tensors, batch_labels)

        def write_to_file(row: pd.Series, stats: FileStats):
            try:
                _tensor = row[contract.CALCULATED_FEATURES_COL_NAME]  # type: torch.FloatTensor
                rnd = random.random()
                if 0 <= rnd < conf.DEFAULT_VALID_DATA_PCT:
                    arr = test_files
                elif conf.DEFAULT_VALID_DATA_PCT <= rnd < conf.DEFAULT_TEST_DATA_PCT:
                    arr = validation_files
                else:
                    arr = train_files

                target_file_idx = random.randint(0, len(arr) - 1)
                arr[target_file_idx].append((_tensor, row[contract.IS_IN_EPITOPE_COL_NAME]))
                if len(arr[target_file_idx]) == max_records_per_df:
                    persist_file_data(arr, target_file_idx)
                    arr[target_file_idx] = []
            except Exception:
                stats.df_errors += 1
                logging.exception("failed handling record")

            stats.handled_records += 1
            if stats.handled_records % 5000 == 0:
                logging.debug(
                    f"handled {stats.handled_records}/{stats.df_records}, failed:{stats.df_errors}")

        for index, file in enumerate(files):
            logging.info(f"reading file ({index + 1}/{len(files)}) {file}")
            stats = FileStats(file)
            df = pd.read_pickle(file)  # type: pd.DataFrame
            stats.df_records = len(df)
            stats.handled_records = 0
            stats.df_errors = 0
            df.apply(lambda x: write_to_file(x, stats), axis=1)
            del df
            if not preserve_files_in_process:
                logging.info(f"deleting file {file}")

        for lst in [train_files, validation_files, test_files]:
            for index in range(len(lst)):
                persist_file_data(lst, index)

        logging.info(
            f"done! created train: {all_stats.created_train_dfs}, validation: {all_stats.created_validation_dfs}, test: {all_stats.created_test_dfs}")
        return train_files_dir, validation_files_dir, test_files_dir
