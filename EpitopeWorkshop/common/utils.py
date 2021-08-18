import time
from typing import List, Optional, Union

from Bio.Seq import Seq


def split_to_subsequences(sequence: Union[str, Seq], size: int, start_index: int = 0,
                          end_index: Optional[int] = None) -> List[Seq]:
    end_index = end_index or len(sequence) - size + 1
    sub_seqs = []
    for current_start_index in range(start_index, end_index):
        sub_seqs.append(sequence[current_start_index:current_start_index + size])
    return sub_seqs
