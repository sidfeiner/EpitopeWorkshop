from typing import Generator

from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from EpitopeWorkshop.common import utils

# Proline and Glycine have poor helix-forming propensities according to
# https://www.sciencedirect.com/topics/chemistry/alpha-helix
NON_HELIX_AMINO_ACIDS = ['G', 'P']


class SecondaryStructurePredictor:
    def __init__(self, min_window: int, max_window: int):
        self.min_window = min_window
        self.max_window = max_window

    def _create_subsequence_generator(self, full_sequence: str, aa_index: int) -> Generator[Seq, None, None]:
        for window_size in range(self.min_window, self.max_window + 1):
            start_index = max(aa_index - window_size // 2, 0)
            end_index = min(aa_index + window_size // 2, len(full_sequence) - window_size // 2 + 1)

            for current_start_index in range(start_index, end_index):
                yield full_sequence[current_start_index:current_start_index + window_size]

    def is_non_alpha_helix_amino_acid(self, aa: str):
        return aa.upper() in NON_HELIX_AMINO_ACIDS

    def predict(self, full_sequence: str, aa_index: int) -> (float, float):
        """
        :param full_sequence: full protein sequence object
        :param aa_index: index of current amino acid we wish to analyze
        :return: Tuple with average probability of being in an alpha-helix, and being in a beta-sheet
        """
        subsequences = self._create_subsequence_generator(full_sequence, aa_index)
        subsequences_cnt = 0
        alpha_helix_sum = beta_sheet_sum = 0
        amino_acid = str(full_sequence[aa_index])

        for subseq in subsequences:
            alpha_proba, beta_proba, turn_proba = ProteinAnalysis(subseq).secondary_structure_fraction()
            alpha_helix_sum += alpha_proba
            beta_sheet_sum += beta_proba
            subsequences_cnt += 1

        alpha_helix_proba = 0 if self.is_non_alpha_helix_amino_acid(amino_acid) else alpha_helix_sum / subsequences_cnt
        beta_sheet_proba = beta_sheet_sum / subsequences_cnt

        return alpha_helix_proba, beta_sheet_proba
