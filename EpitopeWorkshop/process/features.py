import pandas as pd
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from memoization import cached
from quantiprot.metrics.aaindex import get_aa2volume
from EpitopeWorkshop.common.contract import *


class FeatureCalculator:
    # Polarity based on: https://teaching.ncl.ac.uk/bms/wiki/index.php/Amino_acids
    POLAR_AAS = ['R', 'N', 'D', 'C', 'Q', 'E', 'H', 'K', 'S', 'T', 'Y']
    AA_TO_VOLUME_MAPPING = get_aa2volume()

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
        return self.AA_TO_VOLUME_MAPPING(row[TYPE_COL_NAME].upper())

    @cached(custom_key_maker=_key_amino_acid)
    def _calculate_hydrophobicity(self, row: pd.Series):
        return ProtParamData.kd[row[TYPE_COL_NAME]]

    @cached(custom_key_maker=_key_amino_acid)
    def _calculate_polarity(self, row: pd.Series):
        return row[TYPE_COL_NAME] in self.POLAR_AAS

    @cached(custom_key_maker=_key_seq_id_amino)
    def _calculate_relative_surface_accessibility(self, row: pd.Series):
        seq, amino_acid_index = row[SEQ_COL_NAME], row[AMINO_ACID_INDEX_COL_NAME]
        return None

    @cached(custom_key_maker=_key_seq_id_amino)
    def _calculate_secondary_surface(self, row: pd.Series):
        return row[ANALYZED_SEQ_COL_NAME].secondary_structure_fraction()

    def _calculate_type(self, row: pd.Series):
        return row[SEQ_COL_NAME][row[AMINO_ACID_INDEX_COL_NAME]].upper()

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df[ANALYZED_SEQ_COL_NAME] = df.apply(self._calculate_analyzed_sequence, axis=1)

        # Type must be calculated first because it is used in next steps
        df[TYPE_COL_NAME] = df.apply(self._calculate_type, axis=1)

        df[HYDROPHOBICITY_COL_NAME] = df.apply(self._calculate_hydrophobicity, axis=1)
        df[COMPUTED_VOLUME_COL_NAME] = df.apply(self._calculate_computed_volume, axis=1)
        df[POLARITY_COL_NAME] = df.apply(self._calculate_polarity, axis=1)
        df[SS_COL_NAME] = df.apply(self._calculate_secondary_surface, axis=1)
        df[RSA_COL_NAME] = df.apply(self._calculate_relative_surface_accessibility, axis=1)
        return df
