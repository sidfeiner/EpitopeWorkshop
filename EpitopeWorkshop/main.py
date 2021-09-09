import glob
import logging
import os
import shutil
from typing import Optional

import multiprocessing
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.scripts.calculate_features import FileFeatureCalculator
from EpitopeWorkshop.scripts.classify_peptide import PeptideClassifier
from EpitopeWorkshop.scripts.over_balance import OverBalancer
from EpitopeWorkshop.scripts.split_data import ShuffleData
from EpitopeWorkshop.scripts.split_iedb_epitopes import SplitIEDBEpitopes


class PoolWorkerResults:
    def __init__(self):
        self.worker_results = {}

    def callback(self, worker_id: int, res):
        self.worker_results[worker_id] = res


class FullFlow:
    def __init__(self):
        self.splitter = SplitIEDBEpitopes()
        self.features = FileFeatureCalculator()
        self.over_balancer = OverBalancer()
        self.shuffle = ShuffleData()
        self.classify = PeptideClassifier()

    def _run_worker_flow(self, sequences_files_dir: str, total_workers: int, worker_id: int,
                         window_size: int = DEFAULT_WINDOW_SIZE,
                         oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                         oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                         oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX,
                         preserve_files_in_process: bool = DEFAULT_PRESERVE_FILES_IN_PROCESS
                         ):
        file_and_dfs = self.features.calculate_features_dir(
            sequences_files_dir, total_workers, worker_id,
            window_size, preserve_files_in_process
        )

        balanced_dir = os.path.join(sequences_files_dir, 'balanced')
        os.makedirs(balanced_dir, exist_ok=True)
        for file_name, df in file_and_dfs:
            # file_name is the path where `df` is stored
            # `df` is the dataframe with the calculated features
            logging.info(f"balancing file {file_name}")
            balanced_df_path = os.path.join(balanced_dir, os.path.basename(file_name))
            self.over_balancer.over_balance_df(
                df, balanced_df_path, oversampling_change_val_proba,
                oversampling_altercation_pct_min, oversampling_altercation_pct_max
            )
            if not preserve_files_in_process:
                os.remove(file_name)  # Delete file that the df is based on (the one with the calculated features)
        logging.info(f"worker {worker_id} finished calculating features and over balancing files")
        return balanced_dir

    def run_flow(self, sequences_files_dir: str, total_workers: int = 1,
                 split_file_to_parts_amt: Optional[int] = None,
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                 oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                 oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX,
                 preserve_files_in_process: bool = DEFAULT_PRESERVE_FILES_IN_PROCESS
                 ):
        """
        This is the main entry point for learning.
        This will split the original data to files, calculate the features, over balance and then shuffle all
        the data across different files to ensure random order when the model learns.

        :param sequences_files_dir: Directory with fasta files (only `.fasta` files will be handled)
        :param total_workers: Amount of processes to spawn for calculating and overbalancing, defaults to 1
        :param split_file_to_parts_amt: If given, will split every `.fasta` file in the dir to this amount of files
        :param window_size: Window size to analyze every protein sequence. Defaults to 9
        :param oversampling_change_val_proba: Optional. A number between 0 and 1 that affects when a field should be
                                              slightly altercated during over balance. Defaults to 0.2
        :param oversampling_altercation_pct_min: Optional. A number in percentage that decides the lowest bound of altercation
                                                 for a field's value during over balance.. Defaults to 97.
        :param oversampling_altercation_pct_max: Optional. A number in percentage that decides the highest bound of altercation
                                                 for a field's value during over balance.. Defaults to 103.
        :param preserve_files_in_process: If true, all files created during the process will be deleted when they're not
                                          needed anymore.
        """
        if split_file_to_parts_amt is not None:
            sequences_files_dir = os.path.join(sequences_files_dir, 'split')
            logging.info(f'splitting original files into dir: {sequences_files_dir}')
            for file in glob.glob(os.path.join(sequences_files_dir, '*.fasta')):
                self.splitter.split_iebdb_file(file, split_file_to_parts_amt, sequences_files_dir)

        results = PoolWorkerResults()

        with multiprocessing.Pool(total_workers) as pool:
            for i in range(total_workers):
                pool.apply_async(
                    self._run_worker_flow,
                    (
                        sequences_files_dir, total_workers, i,
                        window_size, oversampling_change_val_proba, oversampling_altercation_pct_min,
                        oversampling_altercation_pct_max, preserve_files_in_process
                    ),
                    callback=lambda val: results.callback(i, val)
                )
        pool.join()  # Wait for all data to be balanced

        logging.info("done waiting for all worker threads to calculate features and over balance")
        balanced_dirs = list({balanced_dir for balanced_dir in results.worker_results.values()})

        train_dirs = set()
        validation_dirs = set()
        test_dirs = set()

        logging.info("shuffling data")
        for balanced_dir in balanced_dirs:
            logging.info(f"shuffling data in dir {balanced_dir}")
            train_dir, validation_dir, test_dir = self.shuffle.shuffle_data_dir(balanced_dir)
            train_dirs.add(train_dir)
            validation_dirs.add(validation_dir)
            test_dirs.add(test_dir)
        logging.info("done shuffling data")
