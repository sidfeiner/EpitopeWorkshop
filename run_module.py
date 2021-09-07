import logging
import fire
import pandas as pd
import numpy as np
import torch
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from EpitopeWorkshop.common.conf import DEFAULT_USING_NET_DEVICE, PATH_TO_CNN,\
    DEFAULT_IS_IN_EPITOPE_THRESHOLD, MIN_EPITOP_SIZE
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

    def create_upper_case_string(self, raw_seq: list) -> str:
        start_epitope_seq_index = 0
        in_sequence = 0
        for i, amino_acid in enumerate(raw_seq):
            if amino_acid.islower():  # Not part of the epitope
                if in_sequence >= MIN_EPITOP_SIZE:  # The sequence is long enough to be at the epitope
                    start_epitop_seq_index = 0
                    in_sequence = 0
                else:  # The sequence is not long enough to be part of the epitope
                    while in_sequence > 0:
                        raw_seq[start_epitope_seq_index] = raw_seq[start_epitope_seq_index].lower()
                        in_sequence -= 1
                        start_epitope_seq_index += 1
            else:  # Found capital char
                if in_sequence:  # Were already started epitope sequence
                    in_sequence += 1
                else:  # Starting an epitope sequence
                    start_epitope_seq_index = i
                    in_sequence += 1

        return "".join(raw_seq)

    def make_predication_str(self, peptide: str, epitope_probas: pd.DataFrame,
                             threshold: float = DEFAULT_IS_IN_EPITOPE_THRESHOLD) -> str:
        letters_data = pd.Series([letter.lower() for letter in peptide], name="Letters", dtype="string")
        epitopes_classification = epitope_probas >= threshold
        df = pd.concat([letters_data, epitopes_classification], axis=1)
        df.columns = ['Letters', 'epitopes_classification']
        df['result_seq'] = df['Letters'].mask(df['epitopes_classification'], df['Letters'].str.upper())
        result_seq_lis = df['result_seq'].tolist()

        return self.create_upper_case_string(result_seq_lis)

    def create_head_map(self, epitope_probas):
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
        predication = self.make_predication_str(peptide, epitope_probas)

        print(f"The predicated protein sequence is:\n {predication}")
        self.create_head_map(epitope_probas)

        print("Done")


if __name__ == '__main__':
    fire.Fire((ClassifyPeptide))
