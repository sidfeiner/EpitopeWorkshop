import glob
import logging
import os
import pickle
import re

import pandas as pd
from EpitopeWorkshop.common import contract
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.process.balance.balance import OverSamplingBalancer
from EpitopeWorkshop.process.balance.transform import FeatureTransformer


def print_balanced_data(df: pd.DataFrame):
    vals = df[contract.IS_IN_EPITOPE_COL_NAME].value_counts(normalize=True,
                                                            sort=True)
    print(f"total records: {len(df)}. Frequencies:")
    for val, freq in enumerate(vals):
        print(f"{val}: {freq * 100}%")


class OverBalancer:
    def over_balance_file(self, features_df_pickle_path: str,
                          oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                          oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                          oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX):
        logging.info(f"oversampling file {features_df_pickle_path}")
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

        logging.info("loading df")
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

    def over_balance_dir(self, features_df_pickle_path_dir: str, total_workers: int = 1, worker_id: int = 0,
                         oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                         oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                         oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX):
        logging.info(f"over balancing files in dir: {features_df_pickle_path_dir}")
        files = glob.glob(os.path.join(features_df_pickle_path_dir, '*_features.fasta'))
        files = [file for file in files if
                 int(re.search(r'_(\d+)_features', file).group(1)) % total_workers == worker_id]
        for file in files:
            self.over_balance_file(file, oversampling_change_val_proba, oversampling_altercation_pct_min,
                                   oversampling_altercation_pct_max)
        logging.info("done over balancing all files")
