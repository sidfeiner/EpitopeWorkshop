import random

import numpy as np
import torch

from EpitopeWorkshop.common import contract


class FeatureTransformer:
    def __init__(self, min_pct: int, max_pct: int, change_val_proba: float):
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.change_val_proba = change_val_proba

    def _get_random_pct(self):
        return random.randint(self.min_pct, self.max_pct)

    @staticmethod
    def _set_proba_val(row: np.ndarray, val_index: int, pct_change: int):
        new_val = row[val_index] * (pct_change / 100)
        if new_val > 1:
            new_val = 1
        if new_val < 0:
            new_val = 0
        row[val_index] = new_val

    def _transform_row(self, row: np.ndarray) -> np.ndarray:
        positive_type_features = {type_col for type_col in contract.TYPE_COLUMNS.values() if
                                  row[contract.FEATURES_TO_INDEX_MAPPING[type_col]]}

        # Add extra type
        add_extra_positive_type = random.random() <= self.change_val_proba
        if add_extra_positive_type:
            potential_types = [type_col for type_col in contract.TYPE_COLUMNS.values() if
                               type_col not in positive_type_features]
            random_potential_type = random.choice(potential_types)
            row[contract.FEATURES_TO_INDEX_MAPPING[random_potential_type]] = 1

        # Change Volume
        change_volume = random.random() <= self.change_val_proba
        if change_volume:
            pct_change = self._get_random_pct()
            row[contract.FEATURES_TO_INDEX_MAPPING[contract.COMPUTED_VOLUME_COL_NAME]] *= (pct_change / 100)

        # Change Hydrophobicity
        change_hydro = random.random() <= self.change_val_proba
        if change_hydro:
            pct_change = self._get_random_pct()
            row[contract.FEATURES_TO_INDEX_MAPPING[contract.HYDROPHOBICITY_COL_NAME]] *= (pct_change / 100)

        # Change RSA
        change_rsa = random.random() <= self.change_val_proba
        if change_rsa:
            pct_change = self._get_random_pct()
            row[contract.FEATURES_TO_INDEX_MAPPING[contract.RSA_COL_NAME]] *= (pct_change / 100)

        # Change Secondary Structure Alpha Helix probability
        change_ss_alpha = random.random() <= self.change_val_proba
        if change_ss_alpha:
            pct_change = self._get_random_pct()
            index = contract.FEATURES_TO_INDEX_MAPPING[contract.SS_ALPHA_HELIX_PROBA_COL_NAME]
            self._set_proba_val(row, index, pct_change)

        # Change Secondary Structure Beta Sheet probability
        change_ss_beta = random.random() <= self.change_val_proba
        if change_ss_beta:
            pct_change = self._get_random_pct()
            index = contract.FEATURES_TO_INDEX_MAPPING[contract.SS_ALPHA_HELIX_PROBA_COL_NAME]
            self._set_proba_val(row, index, pct_change)

        # Change Polarity probability
        change_polar_proba = random.random() <= self.change_val_proba
        if change_polar_proba:
            pct_change = self._get_random_pct()
            index = contract.FEATURES_TO_INDEX_MAPPING[contract.POLARITY_PROBA_COL_NAME]
            self._set_proba_val(row, index, pct_change)

        return row

    def transform(self, features: torch.Tensor):
        features_np = features.numpy()
        channels = []
        for channel in features_np:
            channel_rows = []
            for row in channel:
                transformed_row = self._transform_row(row)
                channel_rows.append(transformed_row)
            channels.append(channel_rows)
        return torch.FloatTensor(channels)
