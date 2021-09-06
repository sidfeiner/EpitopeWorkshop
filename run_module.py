import logging
import fire
import pandas as pd
import numpy as np
import torch

from EpitopeWorkshop.common.conf import DEFAULT_USING_NET_DEVICE, PATH_TO_CNN, DEFAULT_IS_IN_EPITOPE_THRESHOLD
from EpitopeWorkshop.cnn.cnn import CNN
from EpitopeWorkshop.dataset.EpitopeDataset import EpitopeDataset
from EpitopeWorkshop.common import contract, utils, conf
from EpitopeWorkshop.process.features import FeatureCalculator
from EpitopeWorkshop.process.read import load_fasta_row_per_window, prep_data_per_sequence, build_df

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

    def classify_peptide(self, peptide: str, threshold: float = DEFAULT_IS_IN_EPITOPE_THRESHOLD,
                         device: str = DEFAULT_USING_NET_DEVICE):
        logging.info(f"preparing data to input")
        data = self.prepare_data(peptide)

        logging.info(f"activating trained CNN")
        cnn = self.load_net()

        logging.info(f"running.....")
        epitope_probas = torch.sigmoid(cnn(data))

        logging.info(f"finished calculating probabilities, creating predication")
        epitopes_classification = epitope_probas >= threshold

        print("Done")


if __name__ == '__main__':
    fire.Fire(ClassifyPeptide)
