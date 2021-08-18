from typing import List, Optional

from Bio import SeqIO, Seq
import pandas as pd

from EpitopeWorkshop.common import utils
from EpitopeWorkshop.common.contract import *
from EpitopeWorkshop.common.conf import DEFAULT_WINDOW_SIZE


def read_fasta(path: str, with_sliding_window: bool = True,
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
            current_sub_seqs = utils.split_to_subsequences(record.seq, sliding_window_size) if with_sliding_window else [
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
                (sub_seq_ix + index for sub_seq_ix, sub_seq in enumerate(current_sub_seqs) for index in range(len(sub_seq))))
            amino_acid_index_in_subseq.extend((index for sub_seq in current_sub_seqs for index in range(len(sub_seq))))
            is_in_epitope.extend((aa.isupper() for sub_seq in current_sub_seqs for aa in sub_seq))
            if len(record.features):
                print(f"found features: {record.features}")
            if limit_sequences_amt is not None and limit_sequences_amt == idx:
                break

    data = {ID_COL_NAME: ids, SEQ_COL_NAME: seqs, AMINO_ACID_SEQ_INDEX_COL_NAME: amino_acid_index_in_seq,
            SUB_SEQ_COL_NAME: sub_seqs, AMINO_ACID_SUBSEQ_INDEX_COL_NAME: amino_acid_index_in_subseq,
            IS_IN_EPITOPE_COL_NAME: is_in_epitope}
    return pd.DataFrame(data)
