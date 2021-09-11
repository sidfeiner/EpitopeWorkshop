import functools
import glob
import logging
import shutil
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
            balance_func = functools.partial(
                self.over_balancer.over_balance_df,
                oversampling_change_val_proba=oversampling_change_val_proba,
                oversampling_altercation_pct_min=oversampling_altercation_pct_min,
                oversampling_altercation_pct_max=oversampling_altercation_pct_max
            )
        else:
            balance_func = functools.partial(self.under_balancer.under_balance_df)

        for file_name, df in file_and_dfs:
            # file_name is the path where `df` is stored
            # `df` is the dataframe with the calculated features
            logging.info(f"balancing file {file_name}")
            balanced_df_path = os.path.join(balanced_dir, os.path.basename(file_name))
            balance_func(df, balanced_df_path)
            if not preserve_files_in_process:
                os.remove(file_name)  # Delete file that the df is based on (the one with the calculated features)
        logging.info(f"worker {worker_id} finished calculating features and over balancing files")
        return {balanced_dir}

    def _move_files_in_dir_to_dir(self, src_dir: str, dst_dir: str):
        files = glob.glob(os.path.join(src_dir, '*'))
        for file in files:
            shutil.move(file, dst_dir)

    def list_cnns(self):
        """List all trained CNN models that exist and can be referenced in our modules"""
        paths = glob.glob(os.path.join(PATH_TO_CNN_DIR, '*'))
        for path in paths:
            print(f"* {os.path.basename(path)}")

    def classify(self, sequence: str, heat_map_name: Optional[str] = None, cnn_name: str = CNN_NAME):
        """Classify a sequence based on some CNN.
        :param sequence: amino acid sequence
        :param heat_map_name: If given, heat map will be saved to this location (container file-system). Be sure to
                              mount this directory to access it from your computer
        :param cnn_name: Name of CNN to use for this classification. To know which cnns are available, run list-cnns command
        """
        return self._classify.classify_peptide(sequence, heat_map_name, cnn_name)

    def run_flow(self, sequences_files_dir: str, total_workers: int = 1,
                 split_file_to_parts_amt: Optional[int] = None,
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 balancing_method: str = DEFAULT_BALANCING_METHOD,
                 oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                 oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                 oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX,
                 preserve_files_in_process: bool = DEFAULT_PRESERVE_FILES_IN_PROCESS,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 batches_until_test: int = DEFAULT_BATCHES_UNTIL_TEST,
                 max_records_in_final_df: int = DEFAULT_RECORDS_IN_FINAL_DF,
                 epochs: int = DEFAULT_EPOCHS,
                 pos_weight: float = DEFAULT_POS_WEIGHT,
                 weight_decay: float = DEFAULT_WEIGHT_DECAY,
                 cnn_name: str = PATH_TO_USER_CNN
                 ):
        """
        This is the main entry point for learning.
        This will split the original data to files, calculate the features, over balance and then shuffle all
        the data across different files to ensure random order when the model learns.

        :param sequences_files_dir: Directory with fasta files (only `.fasta` files will be handled)
        :param total_workers: Amount of processes to spawn for calculating and overbalancing, defaults to 1
        :param split_file_to_parts_amt: If given, will split every `.fasta` file in the dir to this amount of files
        :param window_size: Window size to analyze every protein sequence. Defaults to 9
        :param balancing_method: Balancing method to use. Can be upper/under/none, defaults to under.
        :param oversampling_change_val_proba: Optional. A number between 0 and 1 that affects when a field should be
                                              slightly altercated during over balance. Defaults to 0.2
        :param oversampling_altercation_pct_min: Optional. A number in percentage that decides the lowest bound of altercation
                                                 for a field's value during over balance.. Defaults to 97.
        :param oversampling_altercation_pct_max: Optional. A number in percentage that decides the highest bound of altercation
                                                 for a field's value during over balance.. Defaults to 103.
        :param preserve_files_in_process: If true, all files created during the process will be deleted when they're not
                                          needed anymore.
        :param batch_size: batch size when the CNN learns
        :param batches_until_test: Batches to learn between testing
        :param epochs: epochs to run the data
        :param pos_weight: If given, will give a positive weight to the loss func
        :param weight_decay: regularization parameter, defaults to 0.01
        :param cnn_name: name of the cnn, use this name if you want to classify using your cnn
        """
        assert balancing_method in (
            vars.BALANCING_METHOD_UNDER, vars.BALANCING_METHOD_OVER, vars.BALANCING_METHOD_UNDER)
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
                     balancing_method, oversampling_change_val_proba, oversampling_altercation_pct_min,
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
                preserve_files_in_process
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
                           os.path.join(PATH_TO_CNN_DIR, cnn_name), batch_size,
                           batches_until_test, pos_weight, weight_decay,
                           preserve_files_in_process=preserve_files_in_process)


if __name__ == '__main__':
    log_format = "%(asctime)s : %(threadName)s : %(levelname)s : %(process)d : %(module)s : %(message)s"
    log_level = os.getenv('LOG_LEVEL', 'WARN')
    logging.basicConfig(format=log_format, level=log_level)
    fire.Fire(FullFlow)
