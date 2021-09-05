import logging
import os
from typing import Optional

from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.scripts.calculate_features import FileFeatureCalculator
from EpitopeWorkshop.scripts.over_balance import OverBalancer


class CalculateBalance(FileFeatureCalculator, OverBalancer):
    def run_flow(self, sequences_files_dir: str, total_workers: int, worker_id: int,
                 partitions_amt: int = DEFAULT_PARTITIONS_AMT,
                 with_sliding_window: bool = DEFAULT_WITH_SLIDING_WINDOW,
                 window_size: int = DEFAULT_WINDOW_SIZE, limit_sequences_amt: Optional[int] = None,
                 oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
                 oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
                 oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX
                 ):
        file_and_dfs = self.calculate_features_dir(
            sequences_files_dir, total_workers, worker_id, partitions_amt,
            with_sliding_window, window_size, limit_sequences_amt
        )

        balanced_dir = os.path.join(sequences_files_dir, 'balanced')
        os.makedirs(balanced_dir, exist_ok=True)
        for file_name, df in file_and_dfs:
            logging.info(f"balancing file {file_name}")
            balanced_df_path = os.path.join(balanced_dir, os.path.basename(file_name))
            self.over_balance_df(
                df, balanced_df_path, oversampling_change_val_proba,
                oversampling_altercation_pct_min, oversampling_altercation_pct_max
            )
