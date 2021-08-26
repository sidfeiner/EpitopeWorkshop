import os
from typing import Optional

import fire
import dask.dataframe as dd
from EpitopeWorkshop.process import read, features
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.process.features import FeatureCalculator


def main(sequences_file_path: str, partitions_amt: int = DEFAULT_PARTITIONS_AMT,
         with_sliding_window: bool = DEFAULT_WITH_SLIDING_WINDOW,
         window_size: int = DEFAULT_WINDOW_SIZE, limit_sequences_amt: Optional[int] = None):
    df = read.load_fasta_row_per_window(sequences_file_path, with_sliding_window, window_size, limit_sequences_amt)
    ddf = dd.from_pandas(df, npartitions=partitions_amt)
    calculator = FeatureCalculator()
    df = calculator.calculate_features(ddf)
    print(df[:20])


if __name__ == '__main__':
    fire.Fire(main)
