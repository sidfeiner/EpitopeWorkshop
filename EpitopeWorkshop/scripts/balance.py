import glob
import logging
import multiprocessing
import os
import pickle
import re

import pandas as pd
from EpitopeWorkshop.common import contract, utils
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.process.balance.balance import OverSamplingBalancer, UnderSamplingBalancer
from EpitopeWorkshop.process.balance.transform import FeatureTransformer


def print_balanced_data(df: pd.DataFrame):
    vals = df[contract.IS_IN_EPITOPE_COL_NAME].value_counts(normalize=True,
                                                            sort=True)
    freqs = [f"{val}: {freq * 100:.3f}" for val, freq in enumerate(vals)]
    logging.info(f"total records: {len(df)}. Frequencies: {', '.join(freqs)}")


class OverBalancer:
    """
    Over balances data and removes duplicates
    """

    def over_balance_df(self, df: pd.DataFrame, final_path: str,
                        oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                        oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                        oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX):
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

        logging.info("balancing data")
        df = balancer.balance(df)
        logging.info("done balancing data")
        print_balanced_data(df)

        logging.info("saving balanced to pickle file")
        df.to_pickle(
            path=final_path,
            protocol=pickle.HIGHEST_PROTOCOL
        )

        return final_path, df

    def over_balance_file(self, features_df_pickle_path: str,
                          oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                          oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                          oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX):
        logging.info(f"oversampling file {features_df_pickle_path}")
        dir_path, basename = os.path.split(features_df_pickle_path)
        name, ext = os.path.splitext(basename)
        logging.info("loading df")
        df = pd.read_pickle(features_df_pickle_path)
        balanced_dir = os.path.join(dir_path, 'balanced')
        os.makedirs(balanced_dir, exist_ok=True)
        return self.over_balance_df(
            df, os.path.join(balanced_dir, f"{name}_balanced{ext}"),
            oversampling_change_val_proba, oversampling_altercation_pct_min,
            oversampling_altercation_pct_max
        )

    def _over_balance_dir(self, features_df_pickle_path_dir: str, total_workers: int = 1, worker_id: int = 0,
                          oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                          oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                          oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX):
        logging.info(f"over balancing files in dir: {features_df_pickle_path_dir}")
        files = glob.glob(os.path.join(features_df_pickle_path_dir, '*_features.df'))
        files = [file for file in files if
                 utils.parse_index_from_partial_data_file(file) % total_workers == worker_id]
        over_balanced_files = []
        for file in files:
            over_balanced_path, _ = self.over_balance_file(file, oversampling_change_val_proba,
                                                           oversampling_altercation_pct_min,
                                                           oversampling_altercation_pct_max)
            over_balanced_files.append(over_balanced_path)

        logging.info("done over balancing all files")
        return over_balanced_files

    def over_balance_dir(self, features_df_pickle_path_dir: str, total_workers: int = 1,
                         oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                         oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                         oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX):
        """
        :param features_df_pickle_path_dir: Directory with pickled dataframes with calculated features
        :param total_workers: Amount of parallelization workers
        :param oversampling_change_val_proba: Optional. A number between 0 and 1 that affects when a field should be
                                              slightly altercated during over balance. Defaults to 0.2
        :param oversampling_altercation_pct_min: Optional. A number in percentage that decides the lowest bound of altercation
                                                 for a field's value during over balance.. Defaults to 90.
        :param oversampling_altercation_pct_max: Optional. A number in percentage that decides the highest bound of altercation
                                                 for a field's value during over balance.. Defaults to 100.
        :return: set of newly created over balanced dirs
        """
        logging.info(f"setting up multiprocessing pool with {total_workers} workers")
        with multiprocessing.Pool(total_workers) as pool:
            results = pool.starmap(
                self._over_balance_dir,
                [
                    (features_df_pickle_path_dir, total_workers, worker_id, oversampling_change_val_proba,
                     oversampling_altercation_pct_min, oversampling_altercation_pct_max)
                    for worker_id in range(total_workers)
                ]
            )
        pool.join()
        return set(results)


class UnderBalancer:
    """
    Under balances data
    """

    def under_balance_df(self, df: pd.DataFrame, final_path: str):
        balancer = UnderSamplingBalancer(
            contract.IS_IN_EPITOPE_COL_NAME,
            balances={
                0: 0.5,
                1: 0.5
            }
        )

        logging.info("balancing data")
        df = balancer.balance(df)
        logging.info("done balancing data")
        print_balanced_data(df)

        logging.info("saving balanced to pickle file")
        df.to_pickle(
            path=final_path,
            protocol=pickle.HIGHEST_PROTOCOL
        )

        return final_path, df

    def under_balance_file(self, features_df_pickle_path: str):
        logging.info(f"undersampling file {features_df_pickle_path}")
        dir_path, basename = os.path.split(features_df_pickle_path)
        name, ext = os.path.splitext(basename)
        logging.info("loading df")
        df = pd.read_pickle(features_df_pickle_path)
        balanced_dir = os.path.join(dir_path, 'balanced')
        os.makedirs(balanced_dir, exist_ok=True)
        return self.under_balance_df(df, os.path.join(balanced_dir, f"{name}_balanced{ext}"))

    def _underbalance_dir(self, features_df_pickle_path_dir: str, total_workers: int = 1, worker_id: int = 0):
        logging.info(f"under balancing files in dir: {features_df_pickle_path_dir}")
        files = glob.glob(os.path.join(features_df_pickle_path_dir, '*_features.df'))
        files = [file for file in files if
                 utils.parse_index_from_partial_data_file(file) % total_workers == worker_id]
        under_balanced_files = []
        for file in files:
            new_file_path, _ = self.under_balance_file(file)
            under_balanced_files.append(under_balanced_files)
        logging.info("done under balancing all files")
        return under_balanced_files

    def under_balance_dir(self, features_df_pickle_path_dir: str, total_workers: int = 1,
                          oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                          oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                          oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX):
        """
        :param features_df_pickle_path_dir: Directory with pickled dataframes with calculated features
        :param total_workers: Amount of parallelization workers
        :return: set of newly created underbalanced paths
        """
        logging.info(f"setting up multiprocessing pool with {total_workers} workers")
        with multiprocessing.Pool(total_workers) as pool:
            results = pool.starmap(
                self._underbalance_dir,
                [
                    (features_df_pickle_path_dir, total_workers, worker_id, oversampling_change_val_proba,
                     oversampling_altercation_pct_min, oversampling_altercation_pct_max)
                    for worker_id in range(total_workers)
                ]
            )
        pool.join()
        return set(results)
