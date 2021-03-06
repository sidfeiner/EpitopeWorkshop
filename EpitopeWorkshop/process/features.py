from typing import List

import pandas as pd
import torch
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from memoization import cached
from quantiprot.metrics.aaindex import get_aa2volume
from EpitopeWorkshop.common import vars, conf
from EpitopeWorkshop.common.conf import DEFAULT_NORMALIZE_VOLUME, DEFAULT_NORMALIZE_HYDROPHOBICITY, \
    DEFAULT_NORMALIZE_SURFACE_ACCESSIBILITY
from EpitopeWorkshop.common.contract import *
from EpitopeWorkshop.process.ss_predictor import SecondaryStructurePredictor

GROUPED_AA = {
    'X': None,
    'B': ['D', 'N'],
    'Z': ['E', 'Q'],
    'J': ['L', 'I']
}


def add_group_type_values(d: dict) -> dict:
    new_dict = {k: float(v) for k, v in d.items()}
    for group, aas in GROUPED_AA.items():
        keys = aas if aas is not None else list(d.keys())
        new_dict[group] = sum([float(d[item]) for item in keys]) / len(keys)
    return new_dict


def create_meta():
    base_meta = {
        HYDROPHOBICITY_COL_NAME: 'f8',
        COMPUTED_VOLUME_COL_NAME: 'f8',
        SS_ALPHA_HELIX_PROBA_COL_NAME: 'f8',
        SS_BETA_SHEET_PROBA_COL_NAME: 'f8',
        SA_COL_NAME: 'f8',
        POLARITY_PROBA_COL_NAME: 'f8'
    }
    for col in TYPE_COLUMNS.values():
        base_meta[col] = 'i'

    return base_meta


class FeatureCalculator:
    AA_TO_VOLUME_MAPPING = add_group_type_values(get_aa2volume().mapping)
    VOLUME_MIN, VOLUME_MAX = min(AA_TO_VOLUME_MAPPING.values()), max(AA_TO_VOLUME_MAPPING.values())

    AA_TO_HYDROPHPBICITY_MAPPING = add_group_type_values(ProtParamData.kd)
    HYDROPHOBICITY_MIN, HYDROPHOBICITY_MAX = min(AA_TO_HYDROPHPBICITY_MAPPING.values()), max(
        AA_TO_HYDROPHPBICITY_MAPPING.values())

    AA_TO_SURFACE_ACCESSIBILITY = add_group_type_values(ProtParamData.em)
    SURFACE_ACCESSIBILITY_MIN, SURFACE_ACCESSIBILITY_MAX = min(AA_TO_SURFACE_ACCESSIBILITY.values()), max(
        AA_TO_SURFACE_ACCESSIBILITY.values())

    AA_TO_POLARITY_MAPPING = add_group_type_values(vars.AMINO_ACIDS_POLARITY_MAPPING)

    APPLY_META = create_meta()

    def __init__(self, normalize_volume: bool = DEFAULT_NORMALIZE_VOLUME,
                 normalize_hydrophpbicity: bool = DEFAULT_NORMALIZE_HYDROPHOBICITY,
                 normalize_surface_accessibility: bool = DEFAULT_NORMALIZE_SURFACE_ACCESSIBILITY,
                 ss_prediction_min_window_size: int = 10, ss_prediction_max_window_size: int = 10,
                 ss_prediction_threshold: float = conf.DEFAULT_SS_PREDICTOR_THRESHOLD):
        self.secondary_structure_predictor = SecondaryStructurePredictor(ss_prediction_min_window_size,
                                                                         ss_prediction_max_window_size,
                                                                         threshold=ss_prediction_threshold)
        self.normalize_volume = normalize_volume
        self.normalize_hydrophpbicity = normalize_hydrophpbicity
        self.normalize_surface_accessibility = normalize_surface_accessibility

    def _key_seq_id_amino(self, row: pd.Series):
        return row[ID_COL_NAME], row[AMINO_ACID_SUBSEQ_INDEX_COL_NAME]

    def _key_seq_id(self, row: pd.Series):
        return row[ID_COL_NAME]

    @cached(custom_key_maker=_key_seq_id)
    def _calculate_analyzed_sequence(self, row: pd.Series):
        return ProteinAnalysis(str(row[SEQ_COL_NAME]).upper())

    @cached
    def _calculate_computed_volume(self, aa_type: str):
        val = self.AA_TO_VOLUME_MAPPING[aa_type]
        return (val - self.VOLUME_MIN) / (self.VOLUME_MAX - self.VOLUME_MIN) if not self.normalize_volume else val

    @cached
    def _calculate_hydrophobicity(self, aa_type: str):
        val = self.AA_TO_HYDROPHPBICITY_MAPPING[aa_type]
        return (val - self.HYDROPHOBICITY_MIN) / (
                self.HYDROPHOBICITY_MAX - self.HYDROPHOBICITY_MIN) if not self.normalize_hydrophpbicity else val

    @cached
    def _calculate_polarity(self, aa_type: str) -> float:
        """
        Returns the probability that it is polar
        """
        return self.AA_TO_POLARITY_MAPPING[aa_type]

    @cached
    def _calculate_surface_accessibility(self, aa_type: str):
        val = self.AA_TO_SURFACE_ACCESSIBILITY[aa_type]
        return (val - self.SURFACE_ACCESSIBILITY_MIN) / (
                self.SURFACE_ACCESSIBILITY_MAX - self.SURFACE_ACCESSIBILITY_MIN) if not self.normalize_surface_accessibility else val

    def _calculate_secondary_surface(self, analyzed_seq: str, aa_index: int):
        if analyzed_seq[aa_index] == conf.NO_AMINO_ACID_CHAR:
            return 0, 0
        alpha_proba, beta_proba = self.secondary_structure_predictor.predict(analyzed_seq, aa_index)
        return alpha_proba, beta_proba

    def _calculate_type(self, row: pd.Series):
        return row[SEQ_COL_NAME][row[AMINO_ACID_SEQ_INDEX_COL_NAME]].upper()

    @cached
    def _calculate_all_types(self, aa_type: str):
        if aa_type == 'X':
            return {col_name: 1 for col_name in TYPE_COLUMNS.values()}
        elif aa_type == conf.NO_AMINO_ACID_CHAR:
            return {col_name: 0 for col_name in TYPE_COLUMNS.values()}
        possible_values = GROUPED_AA.get(aa_type, [aa_type])
        return {col_name: int(t in possible_values) for t, col_name in TYPE_COLUMNS.items()}

    def _add_default_value_column(self, df: pd.DataFrame, column_name: str, default_val):
        df[column_name] = default_val

    def calculate_row_features(self, row: pd.Series) -> torch.Tensor:
        sequence = row[SEQ_COL_NAME]
        sub_sequence = row[SUB_SEQ_COL_NAME]
        sub_sequence_index = row[SUB_SEQ_INDEX_START_COL_NAME]
        subseq_features = []  # type: List[torch.Tensor]
        for aa_rel_index, aa in enumerate(sub_sequence):
            aa = aa.upper()
            aa_abs_index = sub_sequence_index + aa_rel_index

            alpha_proba, beta_proba = \
                0, 0 if aa == conf.NO_AMINO_ACID_CHAR else self._calculate_secondary_surface(str(sequence),
                                                                                             aa_abs_index)

            hydro = 0 if aa == conf.NO_AMINO_ACID_CHAR else self._calculate_hydrophobicity(aa)
            volume = 0 if aa == conf.NO_AMINO_ACID_CHAR else self._calculate_computed_volume(aa)
            sa = 0 if aa == conf.NO_AMINO_ACID_CHAR else self._calculate_surface_accessibility(aa)
            polarity_proba = 0 if aa == conf.NO_AMINO_ACID_CHAR else self._calculate_polarity(aa)

            type_features = self._calculate_all_types(aa)
            features = {
                HYDROPHOBICITY_COL_NAME: hydro,
                COMPUTED_VOLUME_COL_NAME: volume,
                SS_ALPHA_HELIX_PROBA_COL_NAME: alpha_proba,
                SS_BETA_SHEET_PROBA_COL_NAME: alpha_proba,
                SA_COL_NAME: sa,
                POLARITY_PROBA_COL_NAME: polarity_proba
            }

            features.update(type_features)
            lst_features = [features[feature] for feature in FEATURES_ORDERED]

            subseq_features.append(torch.FloatTensor(lst_features))

        return torch.stack(subseq_features).unsqueeze(0)  # Add channel dimension

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        calculated_features = df.apply(self.calculate_row_features, axis=1) \
            .rename(CALCULATED_FEATURES_COL_NAME)
        df = pd.concat([df, calculated_features], axis=1)
        return df
