import random
import re
from typing import List, Union

import torch
from Bio.Seq import Seq
import pandas as pd
import numpy as np

from EpitopeWorkshop.common import conf

PARTIAL_DATA_INDEX_PATTERN = re.compile(r'(\d+)[^\\/]+')


def split_to_subsequences(sequence: Union[str, Seq], size: int, empty_char: str = conf.NO_AMINO_ACID_CHAR) -> List[Seq]:
    sub_seqs = []
    for i in range(len(sequence)):
        left = sequence[max(i - 4, 0):min(i, len(sequence))]
        right = sequence[min(len(sequence), i + 1):min(i + 5, len(sequence))]
        subseq = conf.NO_AMINO_ACID_CHAR * (size // 2 - len(left)) + left + sequence[
            i] + right + conf.NO_AMINO_ACID_CHAR * (size // 2 - len(right))
        sub_seqs.append(subseq)
    return sub_seqs


def df_random_split(df: pd.DataFrame, split_pct: float) -> (pd.DataFrame, pd.DataFrame):
    msk = np.random.rand(len(df)) < split_pct
    train = df[msk]
    test = df[~msk]
    return train, test


def series_random_split(series: List[pd.Series], split_pct: float) -> (List[pd.Series], List[pd.Series]):
    msk = np.random.rand(len(series[0])) < split_pct
    train = [ser[msk].reset_index(drop=True) for ser in series]
    test = [ser[~msk].reset_index(drop=True) for ser in series]
    return train, test


def tensor_random_split(tensor: torch.Tensor, split_pct: float) -> (torch.Tensor, torch.Tensor):
    tensor1, tensor2 = [], []
    for item in tensor:
        if random.random() <= split_pct:
            tensor1.append(item)
        else:
            tensor2.append(item)
    return torch.FloatTensor(tensor1), torch.FloatTensor(tensor2)


def parse_index_from_partial_data_file(path: str) -> int:
    match = PARTIAL_DATA_INDEX_PATTERN.search(path)
    return int(match.group(1))
