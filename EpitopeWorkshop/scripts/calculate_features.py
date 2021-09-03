import glob
import logging
import os
import pickle
import re
import shutil
from typing import Optional

from EpitopeWorkshop.common import contract
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.process import read
from EpitopeWorkshop.process.features import FeatureCalculator


class FileFeatureCalculator:
    def calculate_features(self, sequences_file_path: str, partitions_amt: int = DEFAULT_PARTITIONS_AMT,
                           with_sliding_window: bool = DEFAULT_WITH_SLIDING_WINDOW,
                           window_size: int = DEFAULT_WINDOW_SIZE, limit_sequences_amt: Optional[int] = None):
        logging.info(f"handling file {sequences_file_path}")
        dir_path, basename = os.path.split(sequences_file_path)
        name, ext = os.path.splitext(basename)
        final_dir = os.path.join(dir_path, 'features')
        done_dir = os.path.join(dir_path, 'handled')
        os.makedirs(final_dir, exist_ok=True)
        os.makedirs(done_dir, exist_ok=True)

        df = read.load_fasta_row_per_window(sequences_file_path, with_sliding_window, window_size, limit_sequences_amt)

        # ddf = dd.from_pandas(df, npartitions=partitions_amt)
        calculator = FeatureCalculator()

        logging.info("calculating features")
        df = calculator.calculate_features(df)

        logging.info("saving full file to pickle file")
        raw_data_path = os.path.join(final_dir, f"{name}_features{ext}")
        df[[contract.CALCULATED_FEATURES_COL_NAME, contract.IS_IN_EPITOPE_COL_NAME]].to_pickle(
            path=raw_data_path,
            protocol=pickle.HIGHEST_PROTOCOL
        )

        shutil.move(sequences_file_path, done_dir)
        logging.info(f"done handling {sequences_file_path}")

    def calculate_features_dir(self, sequences_files_dir: str, total_workers: int, worker_id: int,
                               partitions_amt: int = DEFAULT_PARTITIONS_AMT,
                               with_sliding_window: bool = DEFAULT_WITH_SLIDING_WINDOW,
                               window_size: int = DEFAULT_WINDOW_SIZE, limit_sequences_amt: Optional[int] = None):
        logging.info(f"calculating features for fasta files in dir {sequences_files_dir}")
        files = glob.glob(os.path.join(sequences_files_dir, '*.fasta'))
        files = [file for file in files if int(re.search(r'_(\d+)\.fasta', file).group(1)) % total_workers == worker_id]
        for file in files:
            self.calculate_features(
                file, partitions_amt, with_sliding_window,
                window_size, limit_sequences_amt
            )

        print("DONE!")
