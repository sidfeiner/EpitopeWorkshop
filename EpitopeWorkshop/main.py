import functools
import glob
import logging
import shutil
import time
from typing import Optional

import multiprocessing

import fire

from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.scripts.calculate_features import FileFeatureCalculator
from EpitopeWorkshop.scripts.classify_peptide import PeptideClassifier
from EpitopeWorkshop.scripts.balance import OverBalancer, UnderBalancer
from EpitopeWorkshop.scripts.shuffle_data import ShuffleData
from EpitopeWorkshop.scripts.split_iedb_epitopes import SplitIEDBEpitopes
from EpitopeWorkshop.scripts.train import Train


class PoolWorkerResults:
    def __init__(self):
        self.worker_results = {}

    def callback(self, worker_id: int, res):
        self.worker_results[worker_id] = res


class FullFlow:
    def __init__(self):
        self.split = SplitIEDBEpitopes()
        self.features = FileFeatureCalculator()
        self.over_balancer = OverBalancer()
        self.under_balancer = UnderBalancer()
        self.shuffler = ShuffleData()
        self.trainer = Train()
        self._classify = PeptideClassifier()

    def _run_worker_flow(self, sequences_files_dir: str, total_workers: int, worker_id: int,
                         window_size: int = DEFAULT_WINDOW_SIZE,
                         balancing_method: str = DEFAULT_BALANCING_METHOD,
                         balancing_positive_freq: float = DEFAULT_BALANCING_POSITIVE_FREQ,
                         oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                         oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                         oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX,
                         preserve_files_in_process: bool = DEFAULT_PRESERVE_FILES_IN_PROCESS
                         ):
        file_and_dfs = self.features._calculate_features_dir(
            sequences_files_dir, total_workers, worker_id,
            window_size, preserve_files_in_process
        )

        parent_dirs = set()
        if balancing_method == vars.BALANCING_METHOD_NONE:
            for file_name, _ in file_and_dfs:
                parent_dirs.add(os.path.dirname(file_name))
            return parent_dirs

        balanced_dir = os.path.join(sequences_files_dir, 'balanced')
        os.makedirs(balanced_dir, exist_ok=True)

        if balancing_method == vars.BALANCING_METHOD_OVER:
            logging.info("over balancing will be applied")
            balance_func = functools.partial(
                self.over_balancer.over_balance_df,
                oversampling_change_val_proba=oversampling_change_val_proba,
                oversampling_altercation_pct_min=oversampling_altercation_pct_min,
                oversampling_altercation_pct_max=oversampling_altercation_pct_max
            )
        elif balancing_method == vars.BALANCING_METHOD_UNDER:
            # No balancing was already treated
            logging.info("under balancing will be applied")
            balance_func = functools.partial(self.under_balancer.under_balance_df)
        else:
            raise RuntimeError(f"unknown balancing method: {balancing_method}")

        for file_name, df in file_and_dfs:
            # file_name is the path where `df` is stored
            # `df` is the dataframe with the calculated features
            logging.info(f"balancing file {file_name}")
            balanced_df_path = os.path.join(balanced_dir, os.path.basename(file_name))
            balance_func(df, balanced_df_path)
            if not preserve_files_in_process:
                os.remove(file_name)  # Delete file that the df is based on (the one with the calculated features)
        logging.info(f"worker {worker_id} finished calculating features and balancing files")
        return {balanced_dir}

    def _move_files_in_dir_to_dir(self, src_dir: str, dst_dir: str):
        files = glob.glob(os.path.join(src_dir, '*'))
        for file in files:
            shutil.move(file, os.path.join(dst_dir, os.path.basename(file)))

    def list_cnns(self):
        """List all trained CNN models that exist and can be referenced in our modules"""
        for cnn_dir in {PATH_TO_CNN_DIR, PATH_TO_USER_CNN_DIR}:
            paths = glob.glob(os.path.join(cnn_dir, '*'))
            for path in paths:
                print(f"* {os.path.basename(path)}")

    def classify(self, sequence: str, heat_map_name: Optional[str] = None, print_proba: bool = DEFAULT_PRINT_PROBA,
                 print_precision: int = DEFAULT_PRINT_PRECISION, cnn_name: str = CNN_NAME):
        """Classify a sequence based on some CNN.
        :param sequence: amino acid sequence
        :param heat_map_name: If given, heat map will be saved to this location (container file-system). Be sure to
                              mount this directory to access it from your computer
        :param print_proba: If true, will print the probabilities of each amino acid, just as the CNN predicted
        :param print_precision: If `print_proba` is true, then the probabilities will be printed with this precision
                                after the decimal point.
        :param cnn_name: Name of CNN to use for this classification. To know which cnns are available, run list-cnns command
        """
        return self._classify.classify_peptide(sequence, heat_map_name, print_proba, print_precision, cnn_name)

    def run_flow(self, sequences_files_dir: str, total_workers: int = 1,
                 split_file_to_parts_amt: Optional[int] = None,
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 normalize_hydrophobicity: bool = DEFAULT_NORMALIZE_HYDROPHOBICITY,
                 normalize_volume: bool = DEFAULT_NORMALIZE_VOLUME,
                 normalize_surface_accessibility: bool = DEFAULT_NORMALIZE_SURFACE_ACCESSIBILITY,
                 balancing_method: str = DEFAULT_BALANCING_METHOD,
                 balancing_positive_freq: float = DEFAULT_BALANCING_POSITIVE_FREQ,
                 oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                 oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                 oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX,
                 preserve_files_in_process: bool = DEFAULT_PRESERVE_FILES_IN_PROCESS,
                 concurrent_train_files_amt: int = DEFAULT_CONCURRENT_TRAIN_FILES_AMT,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 max_records_in_final_df: int = DEFAULT_RECORDS_IN_FINAL_DF,
                 epochs: int = DEFAULT_EPOCHS,
                 pos_weight: float = DEFAULT_POS_WEIGHT,
                 weight_decay: float = DEFAULT_WEIGHT_DECAY,
                 cnn_name: str = USER_CNN_NAME,
                 threshold: float = DEFAULT_IS_IN_EPITOPE_THRESHOLD
                 ):
        """
        This is the main entry point for learning.
        This will split the original data to files, calculate the features, over balance and then shuffle all
        the data across different files to ensure random order when the model learns.

        :param sequences_files_dir: Directory with fasta files (only `.fasta` files will be handled)
        :param total_workers: Amount of processes to spawn for calculating and overbalancing, defaults to 1
        :param split_file_to_parts_amt: If given, will split every `.fasta` file in the dir to this amount of files
        :param window_size: Window size to analyze every protein sequence. Defaults to 9
        :param normalize_hydrophobicity: If true, hydrophobicity values will be normalized during pre-process in CNN
        :param normalize_volume: If true, amino acid volume values will be normalized during pre-process in CNN
        :param normalize_surface_accessibility: If true, amino acid SA values will be normalized during pre-process in CNN
        :param balancing_method: Balancing method to use. Can be upper/under/none, defaults to under
        :param balancing_positive_freq: Number between 0 and 1. This will be the frequency of positive labels in our dataset after balancing
        :param oversampling_change_val_proba: Optional. A number between 0 and 1 that affects when a field should be
                                              slightly altercated during over balance. Defaults to 0.2
        :param oversampling_altercation_pct_min: Optional. A number in percentage that decides the lowest bound of altercation
                                                 for a field's value during over balance.. Defaults to 97.
        :param oversampling_altercation_pct_max: Optional. A number in percentage that decides the highest bound of altercation
                                                 for a field's value during over balance.. Defaults to 103.
        :param preserve_files_in_process: If true, all files created during the process will be deleted when they're not
                                          needed anymore.
        :param concurrent_train_files_amt: Amount of concurrent train files to randomly shuffle the data.
                                           This amount sets the amount of concurrent validation/test files.
        :param batch_size: batch size when the CNN learns
        :param epochs: epochs to run the data
        :param pos_weight: If given, will give a positive weight to the loss func
        :param weight_decay: regularization parameter, defaults to 0.01
        :param cnn_name: name of the cnn, use this name if you want to classify using your cnn
        :param threshold: Threshold that decides if if an amino acid is part of the epitope
        """
        assert balancing_method in (
            vars.BALANCING_METHOD_NONE, vars.BALANCING_METHOD_OVER, vars.BALANCING_METHOD_UNDER
        )
        assert 0 < balancing_positive_freq < 1
        if split_file_to_parts_amt is not None:
            logging.info(f'splitting original files into dir: {sequences_files_dir}')
            files = glob.glob(os.path.join(sequences_files_dir, '*.fasta'))
            sequences_files_dir = os.path.join(sequences_files_dir, 'split')
            os.makedirs(sequences_files_dir, exist_ok=True)
            for file in files:
                sequences_files_dir = self.split.split_iebdb_file(file, split_file_to_parts_amt,
                                                                  sequences_files_dir)
            logging.info('done splitting original files')

        with multiprocessing.Pool(total_workers) as pool:
            results = pool.starmap(
                self._run_worker_flow,
                [
                    (sequences_files_dir, total_workers, worker_id, window_size,
                     balancing_method, balancing_positive_freq, oversampling_change_val_proba,
                     oversampling_altercation_pct_min,
                     oversampling_altercation_pct_max, preserve_files_in_process) for worker_id in
                    range(total_workers)
                ]
            )
        pool.join()  # Wait for all data to be balanced

        logging.info("done waiting for all worker threads to calculate features and over balance")
        balanced_dirs = {balanced_dir for balanced_dirs in results for balanced_dir in balanced_dirs}

        train_dirs = set()
        validation_dirs = set()
        test_dirs = set()

        logging.info("shuffling data")
        for balanced_dir in balanced_dirs:
            logging.info(f"shuffling data in dir {balanced_dir}")
            train_dir, validation_dir, test_dir = self.shuffler.shuffle_data_dir(
                balanced_dir,
                max_records_in_final_df,
                preserve_files_in_process,
                concurrent_train_files_amt
            )
            train_dirs.add(train_dir)
            validation_dirs.add(validation_dir)
            test_dirs.add(test_dir)
        logging.info("done shuffling data")

        all_train_files_dir = os.path.join(sequences_files_dir, 'all-train')
        all_validation_files_dir = os.path.join(sequences_files_dir, 'all-validation')
        all_test_files_dir = os.path.join(sequences_files_dir, 'all-test')
        [
            os.makedirs(path, exist_ok=True)
            for path in [all_train_files_dir, all_validation_files_dir, all_test_files_dir]
        ]

        dir_mapping = {
            all_train_files_dir: train_dirs,
            all_validation_files_dir: validation_dirs,
            all_test_files_dir: test_dirs
        }
        for dst, srcs in dir_mapping.items():
            for src in srcs:
                self._move_files_in_dir_to_dir(src, dst)

        self.trainer.train(all_train_files_dir, all_validation_files_dir, all_test_files_dir, epochs,
                           cnn_name, batch_size,
                           pos_weight, weight_decay,
                           normalize_hydrophobicity, normalize_volume, normalize_surface_accessibility, threshold,
                           preserve_files_in_process=preserve_files_in_process)


if __name__ == '__main__':
    log_format = "%(asctime)s : %(threadName)s : %(levelname)s : %(process)d : %(module)s : %(message)s"
    log_level = os.getenv('LOG_LEVEL', 'WARN')
    logging.basicConfig(format=log_format, level=log_level)
    fire.Fire(FullFlow)
