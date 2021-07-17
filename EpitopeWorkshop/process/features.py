import pandas as pd
from memoization import cached

from EpitopeWorkshop.common.contract import *


def _get_row_memoization_id(row: pd.Series):
    return row[ID_COL_NAME], row[AMINO_ACID_INDEX_COL_NAME]


@cached(custom_key_maker=_get_row_memoization_id)
def _calculate_computed_volume(row: pd.Series):
    return None


@cached(custom_key_maker=_get_row_memoization_id)
def _calculate_hydrophobicity(row: pd.Series):
    return None


@cached(custom_key_maker=_get_row_memoization_id)
def _calculate_polarity(row: pd.Series):
    return None


@cached(custom_key_maker=_get_row_memoization_id)
def _calculate_relative_surface_accessibility(row: pd.Series):
    return None


@cached(custom_key_maker=_get_row_memoization_id)
def _calculate_secondary_surface(row: pd.Series):
    return None


@cached(custom_key_maker=_get_row_memoization_id)
def _calculate_type(row: pd.Series):
    return None


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df[COMPUTED_VOLUME_COL_NAME] = df.apply(_calculate_computed_volume, axis=1)
    df[HYDROPHOBICITY_COL_NAME] = df.apply(_calculate_hydrophobicity, axis=1)
    df[POLARITY_COL_NAME] = df.apply(_calculate_polarity, axis=1)
    df[RSA_COL_NAME] = df.apply(_calculate_relative_surface_accessibility, axis=1)
    df[SS_COL_NAME] = df.apply(_calculate_secondary_surface, axis=1)
    df[TYPE_COL_NAME] = df.apply(_calculate_type, axis=1)
    return df
