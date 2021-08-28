import os
from typing import Optional

import fire
import dask.dataframe as dd
import torch
import torch.nn as nn

from EpitopeWorkshop.cnn.train import training_loop, train
from EpitopeWorkshop.common import contract
from EpitopeWorkshop.dataset.EpitopeDataset import EpitopeDataset
from EpitopeWorkshop.process import read, features
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.process.features import FeatureCalculator
from EpitopeWorkshop.cnn.cnn import CNN

def main(sequences_file_path: str, partitions_amt: int = DEFAULT_PARTITIONS_AMT,
         with_sliding_window: bool = DEFAULT_WITH_SLIDING_WINDOW,
         window_size: int = DEFAULT_WINDOW_SIZE, limit_sequences_amt: Optional[int] = None):
    df = read.load_fasta_row_per_window(sequences_file_path, with_sliding_window, window_size, limit_sequences_amt)
    ddf = dd.from_pandas(df, npartitions=partitions_amt)
    calculator = FeatureCalculator()
    df = calculator.calculate_features(ddf)

    calculated_features = df[contract.CALCULATED_FEATURES]
    labels = df[contract.IS_IN_EPITOPE_COL_NAME]
    ds = EpitopeDataset(calculated_features, labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_train, ds_valid, ds_test = ds.splits()
    print(len(ds_train), len(ds_valid), len(ds_test))

    batch_size = 20
    dl_train, dl_valid, dl_test = ds.iters(batch_size=batch_size)

    cn = CNN()
    a, b, c, d = training_loop(cn, dl_train=dl_train)
    a, b, c, d = train(cn, dl_train=dl_train, dl_test=dl_test)


if __name__ == '__main__':
    fire.Fire(main)