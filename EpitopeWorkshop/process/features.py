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

    def _key_amino_acid(self, row):
        return row[TYPE_COL_NAME]

    def _key_seq_id_amino(self, row: pd.Series):
        return row[ID_COL_NAME], row[AMINO_ACID_INDEX_COL_NAME]

    def _key_seq_id(self, row: pd.Series):
        return row[ID_COL_NAME]

    @cached(custom_key_maker=_key_seq_id)
    def _calculate_analyzed_sequence(self, row: pd.Series):
        return ProteinAnalysis(str(row[SEQ_COL_NAME]).upper())

    @cached(custom_key_maker=_key_amino_acid)
    def _calculate_computed_volume(self, row: pd.Series):
        return self.AA_TO_VOLUME_MAPPING[row[TYPE_COL_NAME].upper()]

    @cached(custom_key_maker=_key_amino_acid)
    def _calculate_hydrophobicity(self, row: pd.Series):
        return self.AA_TO_HYDROPHPBICITY_MAPPING[row[TYPE_COL_NAME]]

    @cached(custom_key_maker=_key_amino_acid)
    def _calculate_polarity(self, row: pd.Series):
        return self.AA_TO_POLARITY_MAPPING[row[TYPE_COL_NAME]]

    @cached(custom_key_maker=_key_amino_acid)
    def _calculate_relative_surface_accessibility(self, row: pd.Series):
        return ProtParamData.em[row[TYPE_COL_NAME]]

    @cached(custom_key_maker=_key_seq_id_amino)
    def _calculate_secondary_surface(self, row: pd.Series):
        return row[ANALYZED_SEQ_COL_NAME].secondary_structure_fraction()

    def _calculate_type(self, row: pd.Series):
        return row[SEQ_COL_NAME][row[AMINO_ACID_INDEX_COL_NAME]].upper()

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # df[ANALYZED_SEQ_COL_NAME] = df.apply(self._calculate_analyzed_sequence, axis=1)

        # Type must be calculated first because it is used in next steps
        df[TYPE_COL_NAME] = df.apply(self._calculate_type, axis=1)

        df[HYDROPHOBICITY_COL_NAME] = df.apply(self._calculate_hydrophobicity, axis=1)
        df[COMPUTED_VOLUME_COL_NAME] = df.apply(self._calculate_computed_volume, axis=1)
        df[POLARITY_COL_NAME] = df.apply(self._calculate_polarity, axis=1)
        # df[SS_COL_NAME] = df.apply(self._calculate_secondary_surface, axis=1)
        df[RSA_COL_NAME] = df.apply(self._calculate_relative_surface_accessibility, axis=1)
        return df
