from typing import List, Optional, Set

from Bio import SeqIO, Seq
import pandas as pd

from EpitopeWorkshop.common import utils
from EpitopeWorkshop.common.contract import *
from EpitopeWorkshop.common.conf import DEFAULT_WINDOW_SIZE


def load_sequences_from_fasta(path: str) -> Set[str]:
    s = set()
    with open(path) as handle:
        for idx, record in enumerate(SeqIO.parse(handle, "fasta")):
            s.add(record.seq)
    return s


def load_fasta_row_per_aa(path: str, with_sliding_window: bool = True,
                          sliding_window_size: int = DEFAULT_WINDOW_SIZE,
                          limit_sequences_amt: Optional[int] = None
                          ) -> pd.DataFrame:
    ids = []
    seqs = []
    sub_seqs = []
    amino_acid_index_in_seq = []
    amino_acid_index_in_subseq = []
    is_in_epitope = []
    with open(path) as handle:
        for idx, record in enumerate(SeqIO.parse(handle, "fasta")):
            current_sub_seqs = utils.split_to_subsequences(record.seq,
                                                           sliding_window_size) if with_sliding_window else [
                record.seq]
            if with_sliding_window:
                assert len(current_sub_seqs) == (len(record.seq) - sliding_window_size + 1)
            mult_basic_values = len(current_sub_seqs) * sliding_window_size if with_sliding_window else len(record.seq)
            mult_subseq_values = sliding_window_size if with_sliding_window else len(record.seq)
            ids.extend(mult_basic_values * [record.id])
            seqs.extend(mult_basic_values * [record.seq])
            sub_seqs.extend((sub_seq for current_sub_seq in current_sub_seqs for sub_seq in
                             [current_sub_seq] * mult_subseq_values))
            amino_acid_index_in_seq.extend(
                (sub_seq_ix + index for sub_seq_ix, sub_seq in enumerate(current_sub_seqs) for index in
                 range(len(sub_seq))))
            amino_acid_index_in_subseq.extend((index for sub_seq in current_sub_seqs for index in range(len(sub_seq))))
            is_in_epitope.extend((int(aa.isupper()) for sub_seq in current_sub_seqs for aa in sub_seq))
            if len(record.features):
                print(f"found features: {record.features}")
            if limit_sequences_amt is not None and limit_sequences_amt == idx:
                break

    data = {ID_COL_NAME: ids, SEQ_COL_NAME: seqs, AMINO_ACID_SEQ_INDEX_COL_NAME: amino_acid_index_in_seq,
            SUB_SEQ_COL_NAME: sub_seqs, AMINO_ACID_SUBSEQ_INDEX_COL_NAME: amino_acid_index_in_subseq,
            IS_IN_EPITOPE_COL_NAME: is_in_epitope}
    return pd.DataFrame(data)


def load_fasta_row_per_window(path: str, with_sliding_window: bool = True,
                              sliding_window_size: int = DEFAULT_WINDOW_SIZE,
                              limit_sequences_amt: Optional[int] = None
                              ) -> pd.DataFrame:
    ids = []
    seqs = []
    sub_seqs = []
    sub_sequence_index_start = []
    is_in_epitope = []  # type: List[List[int]]
    with open(path) as handle:
        for idx, record in enumerate(SeqIO.parse(handle, "fasta")):
            current_sub_seqs = utils.split_to_subsequences(record.seq,
                                                           sliding_window_size) if with_sliding_window else [
                record.seq]
            if with_sliding_window:
                assert len(current_sub_seqs) == (len(record.seq) - sliding_window_size + 1)
            mult_basic_values = len(current_sub_seqs) if with_sliding_window else len(record.seq)
            ids.extend(mult_basic_values * [record.id])
            seqs.extend(mult_basic_values * [record.seq])
            sub_seqs.extend(current_sub_seqs)
            sub_sequence_index_start.extend(range(len(current_sub_seqs)))
            is_in_epitope.extend(([int(aa.isupper()) for aa in sub_seq] for sub_seq in current_sub_seqs))
            if len(record.features):
                print(f"found features: {record.features}")
            if limit_sequences_amt is not None and limit_sequences_amt == idx:
                break

    data = {ID_COL_NAME: ids, SEQ_COL_NAME: seqs, SUB_SEQ_INDEX_START_COL_NAME: sub_sequence_index_start,
            SUB_SEQ_COL_NAME: sub_seqs, IS_IN_EPITOPE_COL_NAME: is_in_epitope, }
    return pd.DataFrame(data)
