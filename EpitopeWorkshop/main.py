import glob
import logging
import os
import pickle
import re
import shutil
from typing import Optional

import fire
import dask.dataframe as dd
import pandas as pd

from EpitopeWorkshop.cnn.train import train_model
from EpitopeWorkshop.common import contract
from EpitopeWorkshop.dataset.EpitopeDataset import EpitopeDataset
from EpitopeWorkshop.process import read
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.process.balance.balance import OverSamplingBalancer, UnderSamplingBalancer
from EpitopeWorkshop.process.balance.transform import FeatureTransformer
from EpitopeWorkshop.process.features import FeatureCalculator
from EpitopeWorkshop.cnn.cnn import CNN

log_format = "%(asctime)s : %(threadName)s: %(levelname)s : %(name)s : %(module)s : %(message)s"
logging.basicConfig(format=log_format, level=logging.DEBUG)


def print_balanced_data(df: pd.DataFrame):
    vals = df[contract.IS_IN_EPITOPE_COL_NAME].value_counts(normalize=True,
                                                            sort=True)
    print(f"total records: {len(df)}. Frequencies:")
    for val, freq in enumerate(vals):
        print(f"{val}: {freq * 100}%")


class Epitopes:

    def process_file(self, sequences_file_path: str, partitions_amt: int = DEFAULT_PARTITIONS_AMT,
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

        ddf = dd.from_pandas(df, npartitions=partitions_amt)
        calculator = FeatureCalculator()

        logging.info("calculating features")
        df = calculator.calculate_features(ddf)

        logging.info("saving full file to pickle file")
        raw_data_path = os.path.join(final_dir, f"{name}_features{ext}")
        df[[contract.CALCULATED_FEATURES_COL_NAME, contract.IS_IN_EPITOPE_COL_NAME]].to_pickle(
            path=raw_data_path,
            protocol=pickle.HIGHEST_PROTOCOL
        )

        shutil.move(sequences_file_path, done_dir)
        logging.info(f"done handling {sequences_file_path}")

    def main(self, sequences_files_dir: str, total_workers: int, worker_id: int,
             partitions_amt: int = DEFAULT_PARTITIONS_AMT,
             with_sliding_window: bool = DEFAULT_WITH_SLIDING_WINDOW,
             window_size: int = DEFAULT_WINDOW_SIZE, limit_sequences_amt: Optional[int] = None,
             oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
             oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
             oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX):
        files = glob.glob(os.path.join(sequences_files_dir, '*.fasta'))
        files = [file for file in files if int(re.search(r'_(\d+)\.fasta', file).group(1)) % total_workers == worker_id]
        for file in files:
            self.process_file(
                file, partitions_amt, with_sliding_window, window_size, limit_sequences_amt,
                oversampling_change_val_proba, oversampling_altercation_pct_min, oversampling_altercation_pct_max
            )

        print("DONE!")
        return

        calculated_features = df[contract.CALCULATED_FEATURES_COL_NAME]
        labels = df[contract.IS_IN_EPITOPE_COL_NAME]
        ds = EpitopeDataset(calculated_features, labels)
        ds_train, ds_valid, ds_test = ds.splits()
        print(len(ds_train), len(ds_valid), len(ds_test))

        dl_train, dl_valid, dl_test = ds.iters(batch_size=DEFAULT_BATCH_SIZE)

        cn = CNN()
        train_model(cn, dl_train=dl_train)
        # a, b, c, d = train(cn, dl_train=dl_train, dl_test=dl_test)

    def over_balance(self, features_df_pickle_path: str,
                     oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                     oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                     oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX):
        logging.info(f"handling file {features_df_pickle_path}")
        dir_path, basename = os.path.split(features_df_pickle_path)
        name, ext = os.path.splitext(basename)

        feature_transformer = FeatureTransformer(
            oversampling_altercation_pct_min,
            oversampling_altercation_pct_max,
            oversampling_change_val_proba
        )
        balancer = OverSamplingBalancer(
            contract.IS_IN_EPITOPE_COL_NAME,
            [(contract.CALCULATED_FEATURES_COL_NAME, feature_transformer.transform)],
            balances={
                0: 0.5,
                1: 0.5
            }
        )

        logging.info("balancing by over sampling")

        df = pd.read_pickle(features_df_pickle_path)
        logging.info("balancing data")
        df = balancer.balance(df)
        logging.info("done balancing data")
        print_balanced_data(df)

        logging.info("saving balanced to pickle file")
        raw_data_path = os.path.join(dir_path, f"{name}_balanced{ext}")
        df.to_pickle(
            path=raw_data_path,
            protocol=pickle.HIGHEST_PROTOCOL
        )


if __name__ == '__main__':
    fire.Fire(Epitopes)
