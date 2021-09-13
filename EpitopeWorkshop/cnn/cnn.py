import os

import torch
import torch.nn as nn

from EpitopeWorkshop.common import conf
from EpitopeWorkshop.common.conf import DEFAULT_NORMALIZE_SURFACE_ACCESSIBILITY, DEFAULT_NORMALIZE_VOLUME, \
    DEFAULT_NORMALIZE_HYDROPHOBICITY
from EpitopeWorkshop.common.contract import *
from EpitopeWorkshop.process.features import FeatureCalculator

KERNEL_SIZE = 3
IN_CHANNELS = 1
LAYER_1_CHANNELS = 8
LAYER_2_OUT_CHANNELS = 16
LAYER_3_OUT_CHANNELS = 25
PADDING = 1
CLASSIFICATION_OPTIONS_AMT = 1


class CNN(nn.Module):
    def __init__(self, normalize_hydrophobicity: bool = DEFAULT_NORMALIZE_HYDROPHOBICITY,
                 normalize_volume: bool = DEFAULT_NORMALIZE_VOLUME,
                 normalize_surface_accessibility: bool = DEFAULT_NORMALIZE_SURFACE_ACCESSIBILITY):
        super().__init__()
        self.normalize_hydrophobicity = normalize_hydrophobicity
        self.normalize_volume = normalize_volume
        self.normalize_surface_accessibility = normalize_surface_accessibility
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                IN_CHANNELS, out_channels=LAYER_1_CHANNELS,
                kernel_size=KERNEL_SIZE, padding=PADDING,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=LAYER_1_CHANNELS, out_channels=LAYER_2_OUT_CHANNELS,
                kernel_size=KERNEL_SIZE),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=LAYER_2_OUT_CHANNELS, out_channels=LAYER_3_OUT_CHANNELS,
                kernel_size=KERNEL_SIZE),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2750, 150),
            nn.ReLU(),
            nn.Linear(150, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, CLASSIFICATION_OPTIONS_AMT),
        )

    def normalize_feature(self, features: torch.Tensor, feat_index: int, min_val: int, max_val: int):
        feature_vals = features[:, feat_index]
        feature_normalized = (feature_vals - min_val) / (max_val - min_val)
        features[:, feat_index] = feature_normalized
        return features

    MIN_HYDRO = min(FeatureCalculator.AA_TO_HYDROPHPBICITY_MAPPING.values())
    MAX_HYDRO = max(FeatureCalculator.AA_TO_HYDROPHPBICITY_MAPPING.values())

    def normalize_hydrophobicity_values(self, batch: torch.Tensor):
        return torch.stack(
            [self.normalize_feature(features[0], FEATURES_TO_INDEX_MAPPING[HYDROPHOBICITY_COL_NAME], self.MIN_HYDRO,
                                    self.MAX_HYDRO).unsqueeze(0) for features in batch])

    MIN_SA = min(FeatureCalculator.AA_TO_SURFACE_ACCESSIBILITY.values())
    MAX_SA = max(FeatureCalculator.AA_TO_SURFACE_ACCESSIBILITY.values())

    def normalize_sa_values(self, batch: torch.Tensor):
        return torch.stack([self.normalize_feature(features[0], FEATURES_TO_INDEX_MAPPING[SA_COL_NAME], self.MIN_SA,
                                                   self.MAX_SA).unsqueeze(0) for features in batch])

    MIN_VOLUME = min(FeatureCalculator.AA_TO_VOLUME_MAPPING.values())
    MAX_VOLUME = max(FeatureCalculator.AA_TO_VOLUME_MAPPING.values())

    def normalize_volume_values(self, batch: torch.Tensor):
        return torch.stack([
            self.normalize_feature(features[0], FEATURES_TO_INDEX_MAPPING[COMPUTED_VOLUME_COL_NAME], self.MIN_VOLUME,
                                   self.MAX_VOLUME).unsqueeze(0) for features in batch])

    def normalize_features(self, batch_features):
        if self.normalize_hydrophobicity:
            batch_features = self.normalize_hydrophobicity_values(batch_features)
        if self.normalize_surface_accessibility:
            batch_features = self.normalize_sa_values(batch_features)
        if self.normalize_volume:
            batch_features = self.normalize_volume_values(batch_features)
        return batch_features

    def forward(self, x):
        features = self.normalize_features(x)
        features = self.feature_extractor(features)
        features = features.view(features.size(0), -1)
        probability = self.classifier(features)
        return probability

    def to_pth(self, path: str):
        torch.save(self.state_dict(), path)

    DIR_PATHS = [conf.PATH_TO_USER_CNN_DIR, conf.PATH_TO_CNN_DIR]

    @classmethod
    def from_pth(cls, cnn_name: str) -> 'CNN':
        cnn = cls()
        for dir_path in cls.DIR_PATHS:
            cnn_path = os.path.join(dir_path, cnn_name)
            if os.path.exists(cnn_path):
                cnn.load_state_dict(torch.load(cnn_path))
                return cnn
        raise FileNotFoundError(f"could not find cnn name {cnn_name}")
