import torch
from torch.utils import data
import pandas as pd


class EpitopeDataset(data.TensorDataset):
    def __init__(self, features: pd.Series, labels: pd.Series, **kwargs):
        """Create an Epitope dataset instance given a pandas Dataframe

        Arguments:
            path: Path to the data file
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        features_ten = torch.stack(features.values.tolist()).to(device)
        labels_ten = torch.FloatTensor(labels.values.tolist()).unsqueeze(1).to(device)
        super(EpitopeDataset, self).__init__(features_ten, labels_ten)
