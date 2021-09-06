import logging
import fire
import pandas as pd
import numpy as np
import torch

import torch.utils.data as data_utils
from EpitopeWorkshop.common.conf import DEFAULT_USING_NET_DEVICE, PATH_TO_CNN, IS_IN_EPITOP_THRESHOLD
from EpitopeWorkshop.cnn.cnn import CNN
from EpitopeWorkshop.dataset.EpitopeDataset import EpitopeDataset
from EpitopeWorkshop.common import contract, utils, conf
from EpitopeWorkshop.process.features import FeatureCalculator
from EpitopeWorkshop.process.read import load_fasta_row_per_window, load_fasta_row_per_aa
from EpitopeWorkshop.scripts.calculate_features import FileFeatureCalculator

log_format = "%(asctime)s : %(threadName)s: %(levelname)s : %(name)s : %(module)s : %(message)s"
logging.basicConfig(format=log_format, level=logging.DEBUG)

def is_upper(prop: torch.float32) -> bool:
    return prop>= IS_IN_EPITOP_THRESHOLD


def load_net(path: str = PATH_TO_CNN):
    cnn = CNN()
    cnn.load_state_dict(torch.load(path, map_location=DEFAULT_USING_NET_DEVICE))
    cnn.eval()

    return cnn

def prepare_data(prot: str) -> EpitopeDataset:
    df = load_fasta_row_per_window(path='C:\\Users\\User\\OneDrive\\TAU\\cs\\EpitopeWorkshop\\iedb_linear_epitopes_partial.fasta')
    calculator = FeatureCalculator()
    data = calculator.calculate_features(df)
    calculated_features = data[contract.CALCULATED_FEATURES_COL_NAME]
    value = [np.array(ten) for ten in calculated_features]
    val = torch.tensor(value, dtype=torch.float32)

    return val


def creat_predication(predications: torch.tensor, letters: pd.Series) -> pd.Series:
    start_seq_index = 0
    end_seq_index = 0
    for i, letter in enumerate(predications):
        if letter >= IS_IN_EPITOP_THRESHOLD:
            if countRowCapital == 0:
                firstCapialIndex = i
            countRowCapital += 1
        else:
            if countRowCapital < conf.MIN_EPITOP_SIZE:
                for x in range(firstCapialIndex, i + 1):
                    finalPredictedSeq[x] = finalPredictedSeq[x].lower()
            firstCapialIndex = i + 1
            countRowCapital = 0

def some_name(input_by_user: str, device: str = DEFAULT_USING_NET_DEVICE):

    letters_data = pd.Series([letter for letter in input_by_user], dtype="string")
    #letters_data.apply(is_upper)
    logging.info(f"preparing data to input")
    data = prepare_data(input_by_user)

    logging.info(f"activating trained CNN")
    cnn = load_net()

    logging.info(f"running.....")
    outputs = pd.DataFrame(cnn(data)).astype("float")
    outputs.apply(is_upper)


    logging.info(f"finished calculating probabilities, creating predication")




    print("Done")

if __name__ == '__main__':
    fire.Fire(some_name)
