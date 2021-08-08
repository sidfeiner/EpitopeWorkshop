import pandas as pd
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from memoization import cached
from quantiprot.metrics.aaindex import get_aa2volume

from EpitopeWorkshop.common import vars
from EpitopeWorkshop.common.contract import *

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

class FeatureCalculator:
    AA_TO_VOLUME_MAPPING = add_group_type_values(get_aa2volume().mapping)
    AA_TO_HYDROPHPBICITY_MAPPING = add_group_type_values(ProtParamData.kd)
    AA_TO_POLARITY_MAPPING = add_group_type_values(vars.AMINO_ACIDS_POLARITY_MAPPING)
    AA_TO_SURFACE_ACCESSIBILITY = add_group_type_values(ProtParamData.em)

    def _key_seq_id_amino(self, row: pd.Series):
        return row[ID_COL_NAME], row[AMINO_ACID_INDEX_COL_NAME]

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

    def _calculate_secondary_surface(self, analyzed_seq):
        return analyzed_seq.secondary_structure_fraction()

    def _calculate_type(self, row: pd.Series):
        return row[SEQ_COL_NAME][row[AMINO_ACID_INDEX_COL_NAME]].upper()

    @cached
    def _calculate_all_types(self, aa_type: str):
        if aa_type == 'X':
            return {col_name: True for col_name in TYPE_COLUMNS.values()}
        possible_values = GROUPED_AA.get(aa_type, [aa_type])
        return {col_name: t in possible_values for t, col_name in TYPE_COLUMNS.items()}

    def _add_default_value_column(self, df: pd.DataFrame, column_name: str, default_val):
        df[column_name] = default_val

    def calculate_row_features(self, row: pd.Series) -> dict:
        aa_type = self._calculate_type(row)
        analyzed_seq = self._calculate_analyzed_sequence(row)
        all_features = {
            ANALYZED_SEQ_COL_NAME: analyzed_seq,
            HYDROPHOBICITY_COL_NAME: self._calculate_hydrophobicity(aa_type),
            COMPUTED_VOLUME_COL_NAME: self._calculate_computed_volume(aa_type),
            SS_COL_NAME: self._calculate_secondary_surface(analyzed_seq),
            RSA_COL_NAME: self._calculate_relative_surface_accessibility(aa_type),
            IS_POLAR_PROBA_COL_NAME: self._calculate_polarity(aa_type)
        }

        type_features = self._calculate_all_types(aa_type)
        all_features.update(type_features)

        return all_features

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        calculated_features = df.apply(self.calculate_row_features, axis=1, result_type='expand')
        df = pd.concat([df, calculated_features], axis=1)
        return df
