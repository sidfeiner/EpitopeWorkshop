from typing import Iterable

import torch
from torch.utils import data
import pandas as pd
import numpy as np
from EpitopeWorkshop.common import utils, conf, contract
from EpitopeWorkshop.common.conf import *


class EpitopeDataset(data.TensorDataset):
    def __init__(self, data: pd.DataFrame, **kwargs):
        """Create an Epitope dataset instance given a pandas Dataframe

        Arguments:
            path: Path to the data file
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        super(EpitopeDataset, self).__init__(*torch.tensor(data.values.astype(np.float32)))
        self.data = data

    def splits(self):
        """Create dataset objects for splits of the Epitope dataset.

        Arguments:
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        df_train, df_rest = utils.df_random_split(self.data, DEFAULT_TRAIN_DATA_PCT)
        valid_pct_in_rest = (len(self.data) * DEFAULT_VALID_DATA_PCT) / len(df_rest)
        df_valid, df_test = utils.df_random_split(df_rest, valid_pct_in_rest)

        return EpitopeDataset(df_train), EpitopeDataset(df_valid), EpitopeDataset(df_test)

    def iters(self, batch_size=32):
        """Create iterator objects for splits of the Epitope dataset.

        Arguments:
            batch_size: Batch_size
        """
        return [torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                            shuffle=True, num_workers=2) for ds in self.splits()]

    def __getitem__(self, index: int):
        return (
            self.tensors[index][:-1],
            self.tensors[index][-1]
        )

    def __len__(self) -> int:
        return len(self.data)
