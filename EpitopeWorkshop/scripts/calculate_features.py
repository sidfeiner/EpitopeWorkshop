import glob
import logging
import multiprocessing
import pickle
import shutil

from EpitopeWorkshop.common import contract, utils
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.process import read
from EpitopeWorkshop.process.features import FeatureCalculator


class FileFeatureCalculator:
    """
    For a given fasta file or directory, calculate the features for every window in every fasta sequence
    """

    def calculate_features(self, sequences_file_path: str,
                           window_size: int = DEFAULT_WINDOW_SIZE,
                           preserve_files_in_process: bool = DEFAULT_PRESERVE_FILES_IN_PROCESS):
        """Calculate the features for every sequence in `sequences_file_path` and persist to disk
        :param sequences_file_path: Path to fasta file
        :param window_size: Window size around every amino acid for which we calculate the features, Defaults to 9
        :param preserve_files_in_process: If False, `sequences_file_path` will be deleted after it's features have
                                          been extracted
        :return: Path to pickled df with calculated features and the dataframe itself
        """
        logging.info(f"calculating features for file {sequences_file_path}")
        dir_path, basename = os.path.split(sequences_file_path)
        name, ext = os.path.splitext(basename)
        final_dir = os.path.join(dir_path, 'features')
        done_dir = os.path.join(dir_path, 'handled')
        os.makedirs(final_dir, exist_ok=True)
        os.makedirs(done_dir, exist_ok=True)

        df = read.load_fasta_row_per_window(sequences_file_path, window_size)

        calculator = FeatureCalculator()

        logging.info("calculating features")
        df = calculator.calculate_features(df)

        logging.info("saving full file to pickle file")
        raw_data_path = os.path.join(final_dir, f"{name}_features{ext}")
        df[[contract.CALCULATED_FEATURES_COL_NAME, contract.IS_IN_EPITOPE_COL_NAME]].to_pickle(
            path=raw_data_path,
            protocol=pickle.HIGHEST_PROTOCOL
        )
        logging.info(f"done handling {sequences_file_path}")
        if preserve_files_in_process:
            shutil.move(sequences_file_path, os.path.join(done_dir, os.path.basename(sequences_file_path)))
        else:
            logging.info(f'removing file {sequences_file_path}')
            os.remove(sequences_file_path)
        return raw_data_path, df

    def _calculate_features_dir(self, sequences_files_dir: str, total_workers: int = 1, worker_id: int = 0,
                                window_size: int = DEFAULT_WINDOW_SIZE,
                                preserve_files_in_process: bool = DEFAULT_PRESERVE_FILES_IN_PROCESS):
        logging.info(f"calculating features for fasta files in dir {sequences_files_dir}")
        files = glob.glob(os.path.join(sequences_files_dir, '*.fasta'))
        files = [file for file in files if utils.parse_index_from_partial_data_file(file) % total_workers == worker_id]
        for file in files:
            yield self.calculate_features(
                file,
                window_size,
                preserve_files_in_process
            )

        logging.info(f"worker {worker_id} is done calculating features dir")

    def calculate_features_dir(self, sequences_files_dir: str, total_workers: int = 1,
                               window_size: int = DEFAULT_WINDOW_SIZE,
                               preserve_files_in_process: bool = DEFAULT_PRESERVE_FILES_IN_PROCESS):
        """Calculate the features for every sequence files inside `sequences_files_dir` and persist to disk
        :param sequences_files_dir: Path to dir with fasta files
        :param total_workers:
        :param window_size: Window size around every amino acid for which we calculate the features, Defaults to 9
        :param preserve_files_in_process: If False, `sequences_file_path` will be deleted after it's features have
               been extracted
        :return: Path to pickled df with calculated features and the dataframe itself
        """
        with multiprocessing.Pool(total_workers) as pool:
            results = pool.starmap(
                self._calculate_features_dir,
                [
                    (sequences_files_dir, total_workers, worker_id, window_size, preserve_files_in_process) for
                    worker_id in range(total_workers)
                ]
            )
        pool.join()  # Wait for all data to be balanced
        return set(results)
