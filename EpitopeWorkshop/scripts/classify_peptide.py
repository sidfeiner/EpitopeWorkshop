import logging
import os

import regex as re
from typing import List, Optional

import fire
import pandas as pd
import torch
import numpy as np
import seaborn as sb

from EpitopeWorkshop.common.conf import DEFAULT_USING_NET_DEVICE, PATH_TO_CNN, \
    DEFAULT_IS_IN_EPITOPE_THRESHOLD, DEFAULT_MIN_EPITOPE_SIZE, CNN_NAME, PATH_TO_CNN_DIR, HEAT_MAP_DIR
from EpitopeWorkshop.cnn.cnn import CNN
from EpitopeWorkshop.common import contract
from EpitopeWorkshop.process.features import FeatureCalculator
from EpitopeWorkshop.process.read import prep_data_per_sequence, build_df


class PeptideClassifier:

    def _load_net(self, cnn_name: str = CNN_NAME):
        cnn = CNN.from_pth(cnn_name)
        cnn.eval()
        return cnn

    def _prepare_data(self, seq: str):
        data = prep_data_per_sequence(seq)
        df = build_df(*data)
        calculator = FeatureCalculator()
        df = calculator.calculate_features(df)
        values = df[contract.CALCULATED_FEATURES_COL_NAME].to_numpy()
        return torch.tensor(np.stack(values))

    def _ensure_valid_epitope_lengths(self, peptide: List[str], min_epitope_size: int = DEFAULT_MIN_EPITOPE_SIZE):
        """Lower epitope sequences that are shorter than min_epitope_size"""
        pattern = re.compile(f"(?<=^|[a-z])(?P<epitope>[A-Z]{{1,{min_epitope_size - 1}}})(?=$|[a-z])")
        peptide_cp = peptide.copy()
        matches = pattern.finditer(''.join(peptide_cp))
        for match in matches:
            for i in range(*match.span('epitope')):
                peptide_cp[i] = peptide_cp[i].lower()
        return ''.join(peptide_cp)

    def _make_prediction_str(self, peptide: str, epitope_probas: pd.DataFrame,
                             threshold: float = DEFAULT_IS_IN_EPITOPE_THRESHOLD) -> str:
        letters_data = pd.Series([letter.lower() for letter in peptide], name="letters", dtype="string")
        epitopes_classification = epitope_probas >= threshold
        df = pd.DataFrame(data={'letters': letters_data,
                                'epitopes_classification': epitopes_classification[0]}
                          )
        df.columns = ['letters', 'epitopes_classification']
        df['result_seq'] = df['letters'].mask(df['epitopes_classification'], df['letters'].str.upper())
        result_seq_lis = df['result_seq'].tolist()

        return self._ensure_valid_epitope_lengths(result_seq_lis)

    def _create_heat_map(self, epitope_probas, path: str):
        heat_map = sb.heatmap(epitope_probas, cmap="YlGnBu")
        figure = heat_map.get_figure()
        figure.savefig(os.path.join(HEAT_MAP_DIR, path))

    def classify_peptide(self, sequence: str, heat_map_name: Optional[str] = None, cnn_name: str = CNN_NAME):
        """
        :param sequence: amino acid sequence
        :param heat_map_name: If given, heat map will be saved to this location (container file-system). Be sure to
                              mount this directory to access it from your computer
        :param cnn_name: Name of CNN to use for this classification
        """
        logging.info(f"preparing data to input")
        data = self._prepare_data(sequence)

        logging.info(f"activating trained CNN")
        cnn = self._load_net(cnn_name)

        logging.info(f"running.....")
        epitope_probas = pd.DataFrame(torch.sigmoid(cnn(data))).astype("float")

        logging.info(f"finished calculating probabilities, creating prediction")
        prediction = self._make_prediction_str(sequence, epitope_probas)

        if heat_map_name is not None:
            self._create_heat_map(epitope_probas, heat_map_name)
        return prediction


if __name__ == '__main__':
    fire.Fire(PeptideClassifier)
