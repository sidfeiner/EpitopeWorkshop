from typing import List, Optional

from Bio import SeqIO, Seq
import pandas as pd
from EpitopeWorkshop.common.contract import *
from EpitopeWorkshop.common.conf import DEFAULT_WINDOW_SIZE


def split_to_subsequences(sequence: Seq, size: int) -> List[str]:
    sub_seqs = []
    for start_index in range(len(sequence) - size + 1):
        sub_seqs.append(sequence[start_index:start_index + size])
    return sub_seqs


def read_fasta(path: str, with_sliding_window: bool = True,
               sliding_window_size: int = DEFAULT_WINDOW_SIZE,
               limit_sequences_amt: Optional[int] = None
               ) -> pd.DataFrame:
    ids = []
    seqs = []
    sub_seqs = []
    amino_acid_index_per_subseq = []
    with open(path) as handle:
        for idx, record in enumerate(SeqIO.parse(handle, "fasta")):
            current_sub_seqs = split_to_subsequences(record.seq, sliding_window_size) if with_sliding_window else [
                record.seq]
            mult_basic_values = len(current_sub_seqs) * sliding_window_size * len(
                record.seq) if with_sliding_window else len(record.seq)
            mult_subseq_values = sliding_window_size * len(record.seq) if with_sliding_window else len(record.seq)
            ids.extend(mult_basic_values * [record.id])
            seqs.extend(mult_basic_values * [record.seq])
            sub_seqs.extend([sub_seq for current_sub_seq in current_sub_seqs for sub_seq in
                             [current_sub_seq] * mult_subseq_values])
            amino_acid_index_per_subseq.extend([index for sub_seq in current_sub_seqs for index in range(len(sub_seq))])
            if len(record.features):
                print(f"found features: {record.features}")
            if limit_sequences_amt is not None and limit_sequences_amt == idx:
                break

    data = {ID_COL_NAME: ids, SEQ_COL_NAME: seqs, SUB_SEQ_COL_NAME: sub_seqs,
            AMINO_ACID_INDEX_COL_NAME: amino_acid_index_per_subseq}
    return pd.DataFrame(data)
