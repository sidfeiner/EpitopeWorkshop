import time
from typing import List, Optional, Union
from Bio.Seq import Seq
import pandas as pd
import numpy as np


def split_to_subsequences(sequence: Union[str, Seq], size: int, start_index: int = 0,
                          end_index: Optional[int] = None) -> List[Seq]:
    end_index = end_index or len(sequence) - size + 1
    sub_seqs = []
    for current_start_index in range(start_index, end_index):
        sub_seqs.append(sequence[current_start_index:current_start_index + size])
    return sub_seqs


def df_random_split(df: pd.DataFrame, split_pct: float) -> (pd.DataFrame, pd.DataFrame):
    msk = np.random.rand(len(df)) < split_pct
    train = df[msk]
    test = df[~msk]
    return train, test
