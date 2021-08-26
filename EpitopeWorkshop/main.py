import os
from typing import Optional

import fire
import dask.dataframe as dd
import torch
from EpitopeWorkshop.common import contract
from EpitopeWorkshop.dataset.EpitopeDataset import EpitopeDataset
from EpitopeWorkshop.process import read, features
from EpitopeWorkshop.common.conf import *
from EpitopeWorkshop.process.features import FeatureCalculator
from EpitopeWorkshop.cnn.cnn import CNN
import torch.optim as optim
import torch.nn as nn
from EpitopeWorkshop.common.contract import NETWORK_INPUT_ARGS


def main(sequences_file_path: str, partitions_amt: int = DEFAULT_PARTITIONS_AMT,
         with_sliding_window: bool = DEFAULT_WITH_SLIDING_WINDOW,
         window_size: int = DEFAULT_WINDOW_SIZE, limit_sequences_amt: Optional[int] = None):
    df = read.load_fasta_row_per_window(sequences_file_path, with_sliding_window, window_size, limit_sequences_amt)
    ddf = dd.from_pandas(df, npartitions=partitions_amt)
    calculator = FeatureCalculator()
    df = calculator.calculate_features(ddf)
    print(df[:20])

    sub_df = df[contract.NETWORK_LABELED_INPUT_ARGS]
    labels_df = df[contract.IS_IN_EPITOPE_COL_NAME]
    ds = EpitopeDataset(sub_df)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_train, ds_valid, ds_test = ds.splits()
    print(len(ds_train), len(ds_valid), len(ds_test))

    dl_train, dl_valid, dl_test = ds.iters(batch_size=26)

    x0, y0 = ds[0]

    print(len(NETWORK_INPUT_ARGS))

    cn = CNN(len(NETWORK_INPUT_ARGS), 26, 1, 5, 1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cn.parameters(), lr=0.001, momentum=0.9)
    print(type(cn))

    a, b, c, d = cn.training_loop(dl_train=dl_train)


if __name__ == '__main__':
    fire.Fire(main)
