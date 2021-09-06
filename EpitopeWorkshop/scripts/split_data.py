import glob
import logging
import os
import random
from typing import List, Tuple, Dict
import pandas as pd

import torch

from EpitopeWorkshop.common import contract


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


class SplitData:
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

    def split_data(self, balanced_files_dir: str, max_records_per_df: int = 5000):
        files = glob.glob(os.path.join(balanced_files_dir, '*.fasta'))
        train_files = [[] for _ in range(10)]  # type: List[List[torch.FloatTensor, int]]
        validation_files = [[] for _ in range(10)]  # type: List[List[torch.FloatTensor, int]]
        test_files = [[] for _ in range(50)]  # type: List[List[torch.FloatTensor, int]]
        hashes = {}  # type: Dict[Tuple[str, str], List[List[torch.FloatTensor, int]]]  # Map a hash to it's destination kind

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
                                         f"iedb_linear_epitopes_{file_idx}_balanced.df")
            self.data_to_df_disk(final_df_path, batch_tensors, batch_labels)

        def write_to_file(row: pd.Series, stats: FileStats):
            try:
                _tensor = row[contract.CALCULATED_FEATURES_COL_NAME]  # type: torch.FloatTensor
                hash_key = (row[contract.ID_COL_NAME], row[contract.SUB_SEQ_COL_NAME])
                if hash_key not in hashes:
                    rnd = random.random()
                    if 0 < rnd < 0.2:
                        hashes[hash_key] = test_files
                    elif 0.2 <= rnd < 0.3:
                        hashes[hash_key] = validation_files
                    else:
                        hashes[hash_key] = train_files

                target_file_idx = random.randint(0, len(hashes[hash_key]) - 1)
                hashes[hash_key][target_file_idx].append((_tensor, row[contract.IS_IN_EPITOPE_COL_NAME]))
                if len(hashes[hash_key][target_file_idx]) == max_records_per_df:
                    persist_file_data(hashes[hash_key], target_file_idx)
                    hashes[hash_key][target_file_idx] = []
            except Exception as e:
                stats.df_errors += 1
                logging.exception("failed handling record")

            stats.handled_records += 1
            if stats.handled_records % 5000 == 0:
                logging.debug(
                    f"handled {stats.handled_records}/{stats.df_records}, failed:{stats.df_errors}")

        for file in files:
            logging.info(f"reading file {file}")
            stats = FileStats(file)
            df = pd.read_pickle(file)  # type: pd.DataFrame
            stats.df_records = len(df)
            stats.handled_records = 0
            stats.df_errors = 0
            df.apply(lambda x: write_to_file(x, stats), axis=1)

        for lst in [train_files, validation_files, test_files]:
            for index in range(len(lst)):
                persist_file_data(lst, index)

        logging.info(
            f"done! created train: {all_stats.created_train_dfs}, validation: {all_stats.created_validation_dfs}, test: {all_stats.created_test_dfs}")
