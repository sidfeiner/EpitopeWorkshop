from typing import List

import pandas as pd
import torch
from Bio.Seq import Seq
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from memoization import cached
from quantiprot.metrics.aaindex import get_aa2volume
import dask.dataframe as dd
from EpitopeWorkshop.common import vars, conf, utils
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
        SS_BETA_SHEET_PROBACOL_NAME: 'f8',
        RSA_COL_NAME: 'f8',
        IS_POLAR_PROBA_COL_NAME: 'f8'
    }
    for col in TYPE_COLUMNS.values():
        base_meta[col] = 'i'

    return base_meta


class FeatureCalculator:
    AA_TO_VOLUME_MAPPING = add_group_type_values(get_aa2volume().mapping)
    AA_TO_HYDROPHPBICITY_MAPPING = add_group_type_values(ProtParamData.kd)
    AA_TO_POLARITY_MAPPING = add_group_type_values(vars.AMINO_ACIDS_POLARITY_MAPPING)
    AA_TO_SURFACE_ACCESSIBILITY = add_group_type_values(ProtParamData.em)

    APPLY_META = create_meta()

    def __init__(self, ss_prediction_min_window_size: int = 10, ss_prediction_max_window_size: int = 10,
                 ss_prediction_threshold: float = conf.DEFAULT_SS_PREDICTOR_THRESHOLD):
        self.secondary_structure_predictor = SecondaryStructurePredictor(ss_prediction_min_window_size,
                                                                         ss_prediction_max_window_size,
                                                                         threshold=ss_prediction_threshold)

    def _key_seq_id_amino(self, row: pd.Series):
        return row[ID_COL_NAME], row[AMINO_ACID_SUBSEQ_INDEX_COL_NAME]

    def _key_seq_id(self, row: pd.Series):
        return row[ID_COL_NAME]

    @cached(custom_key_maker=_key_seq_id)
    def _calculate_analyzed_sequence(self, row: pd.Series):
        return ProteinAnalysis(str(row[SEQ_COL_NAME]).upper())

    @cached
    def _calculate_computed_volume(self, aa_type: str):
        return self.AA_TO_VOLUME_MAPPING[aa_type]

    @cached
    def _calculate_hydrophobicity(self, aa_type: str):
        return self.AA_TO_HYDROPHPBICITY_MAPPING[aa_type]

    @cached
    def _calculate_polarity(self, aa_type: str) -> float:
        """
        Returns the probability that it is polar
        """
        return self.AA_TO_POLARITY_MAPPING[aa_type]

    @cached
    def _calculate_relative_surface_accessibility(self, aa_type: str):
        """NOT GOOD ENOUGH, FIX!"""
        return self.AA_TO_SURFACE_ACCESSIBILITY[aa_type]

    def _calculate_secondary_surface(self, analyzed_seq: str, aa_index: int):
        alpha_proba, beta_proba = self.secondary_structure_predictor.predict(analyzed_seq, aa_index)
        return alpha_proba, beta_proba

    def _calculate_type(self, row: pd.Series):
        return row[SEQ_COL_NAME][row[AMINO_ACID_SEQ_INDEX_COL_NAME]].upper()

    @cached
    def _calculate_all_types(self, aa_type: str):
        if aa_type == 'X':
            return {col_name: 1 for col_name in TYPE_COLUMNS.values()}
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
            alpha_proba, beta_proba = self._calculate_secondary_surface(str(sequence), aa_abs_index)

            type_features = self._calculate_all_types(aa)
            features = {
                HYDROPHOBICITY_COL_NAME: self._calculate_hydrophobicity(aa),
                COMPUTED_VOLUME_COL_NAME: self._calculate_computed_volume(aa),
                SS_ALPHA_HELIX_PROBA_COL_NAME: alpha_proba,
                SS_BETA_SHEET_PROBACOL_NAME: alpha_proba,
                RSA_COL_NAME: self._calculate_relative_surface_accessibility(aa),
                IS_POLAR_PROBA_COL_NAME: self._calculate_polarity(aa)
            }

            features.update(type_features)
            lst_features = [features[feature] for feature in FEATURES_ORDER]

            subseq_features.append(torch.FloatTensor(lst_features))

        return torch.stack(subseq_features)

    def calculate_features(self, ddf: dd.DataFrame) -> dd.DataFrame:
        ddf[IS_IN_EPITOPE_COL_NAME] = ddf[IS_IN_EPITOPE_COL_NAME].apply(torch.LongTensor, meta=('O'))
        calculated_features = ddf.apply(self.calculate_row_features, axis=1, meta=('O')).rename(CALCULATED_FEATURES)
        ddf = dd.concat([ddf, calculated_features], axis=1)
        return ddf.compute()
