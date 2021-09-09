import logging
import regex as re
from typing import List

import fire
import pandas as pd
import torch
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from EpitopeWorkshop.common.conf import DEFAULT_USING_NET_DEVICE, PATH_TO_CNN, \
    DEFAULT_IS_IN_EPITOPE_THRESHOLD, DEFAULT_MIN_EPITOPE_SIZE
from EpitopeWorkshop.cnn.cnn import CNN
from EpitopeWorkshop.common import contract
from EpitopeWorkshop.process.features import FeatureCalculator
from EpitopeWorkshop.process.read import prep_data_per_sequence, build_df

log_format = "%(asctime)s : %(threadName)s: %(levelname)s : %(name)s : %(module)s : %(message)s"
logging.basicConfig(format=log_format, level=logging.DEBUG)


class ClassifyPeptide:

    def load_net(self, path: str = PATH_TO_CNN):
        cnn = CNN()
        cnn.load_state_dict(torch.load(path, map_location=DEFAULT_USING_NET_DEVICE))
        cnn.eval()

        return cnn

    def prepare_data(self, seq: str):
        data = prep_data_per_sequence(seq)
        df = build_df(*data)
        calculator = FeatureCalculator()
        df = calculator.calculate_features(df)
        values = df[contract.CALCULATED_FEATURES_COL_NAME].to_numpy()
        return torch.tensor(np.stack(values))

    def ensure_valid_epitope_lengths(self, peptide: List[str], min_epitope_size: int = DEFAULT_MIN_EPITOPE_SIZE):
        pattern = re.compile(f"(?<=^|[a-z])(?P<epitope>[A-Z]{{1,{min_epitope_size - 1}}})(?=$|[a-z])")
        peptide_cp = peptide.copy()
        matches = pattern.finditer(''.join(peptide_cp))
        for match in matches:
            for i in range(*match.span('epitope')):
                peptide_cp[i] = peptide_cp[i].lower()
        return ''.join(peptide_cp)

    def make_prediction_str(self, peptide: str, epitope_probas: pd.DataFrame,
                            threshold: float = DEFAULT_IS_IN_EPITOPE_THRESHOLD) -> str:
        letters_data = pd.Series([letter.lower() for letter in peptide], name="letters", dtype="string")
        epitopes_classification = epitope_probas >= threshold
        df = pd.DataFrame(data={'letters': letters_data,
                                'epitopes_classification': epitopes_classification[0]}
                          )
        df.columns = ['letters', 'epitopes_classification']
        df['result_seq'] = df['letters'].mask(df['epitopes_classification'], df['letters'].str.upper())
        result_seq_lis = df['result_seq'].tolist()

        return self.ensure_valid_epitope_lengths(result_seq_lis)

    def create_heat_map(self, epitope_probas):
        heat_map = sb.heatmap(epitope_probas, cmap="YlGnBu")
        plt.show()

    def classify_peptide(self, peptide: str, device: str = DEFAULT_USING_NET_DEVICE):
        logging.info(f"preparing data to input")
        data = self.prepare_data(peptide)

        logging.info(f"activating trained CNN")
        cnn = self.load_net()

        logging.info(f"running.....")
        epitope_probas = pd.DataFrame(torch.sigmoid(cnn(data))).astype("float")

        logging.info(f"finished calculating probabilities, creating predication")
        predication = self.make_prediction_str(peptide, epitope_probas)

        print(f"The predicted protein sequence is:\n {predication}")
        self.create_heat_map(epitope_probas)

        print("Done")


if __name__ == '__main__':
    fire.Fire((ClassifyPeptide))
