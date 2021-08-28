from typing import Optional

import fire
import dask.dataframe as dd
import pandas as pd

from EpitopeWorkshop.cnn.train import train_model
from EpitopeWorkshop.common import contract
from EpitopeWorkshop.dataset.EpitopeDataset import EpitopeDataset
from EpitopeWorkshop.process import read
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.process.balance.balance import DataBalancer, UnderSamplingBalancer, OverSamplingBalancer
from EpitopeWorkshop.process.balance.transform import FeatureTransformer
from EpitopeWorkshop.process.features import FeatureCalculator
from EpitopeWorkshop.cnn.cnn import CNN


def print_balanced_data(df: pd.DataFrame):
    vals = df[contract.IS_IN_EPITOPE_COL_NAME].apply(lambda x: x[len(x) // 2].item()).value_counts(normalize=True,
                                                                                                   sort=True)
    print(f"total records: {len(df)}. Frequencies:")
    for val, freq in enumerate(vals):
        print(f"{val}: {freq * 100}%")


def main(sequences_file_path: str, partitions_amt: int = DEFAULT_PARTITIONS_AMT,
         with_sliding_window: bool = DEFAULT_WITH_SLIDING_WINDOW,
         window_size: int = DEFAULT_WINDOW_SIZE, limit_sequences_amt: Optional[int] = None,
         oversampling_change_val_proba: float = DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA,
         oversampling_altercation_pct_min: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN,
         oversampling_altercation_pct_max: int = DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX):
    df = read.load_fasta_row_per_window(sequences_file_path, with_sliding_window, window_size, limit_sequences_amt)

    ddf = dd.from_pandas(df, npartitions=partitions_amt)
    calculator = FeatureCalculator()
    df = calculator.calculate_features(ddf)

    feature_transformer = FeatureTransformer(
        oversampling_altercation_pct_min,
        oversampling_altercation_pct_max,
        oversampling_change_val_proba
    )

    balancer = OverSamplingBalancer(
        contract.IS_IN_EPITOPE_COL_NAME,
        [(contract.CALCULATED_FEATURES_COL_NAME, feature_transformer.transform)],
        transform_val_func=lambda x: x[len(x) // 2].item(),
        balances={
            0: 0.5,
            1: 0.5
        }
    )
    df = balancer.balance(df)
    print_balanced_data(df)

    calculated_features = df[contract.CALCULATED_FEATURES_COL_NAME]
    labels = df[contract.IS_IN_EPITOPE_COL_NAME]
    ds = EpitopeDataset(calculated_features, labels)
    ds_train, ds_valid, ds_test = ds.splits()
    print(len(ds_train), len(ds_valid), len(ds_test))

    dl_train, dl_valid, dl_test = ds.iters(batch_size=DEFAULT_BATCH_SIZE)

    cn = CNN()
    train_model(cn, dl_train=dl_train)
    # a, b, c, d = train(cn, dl_train=dl_train, dl_test=dl_test)


if __name__ == '__main__':
    fire.Fire(main)
