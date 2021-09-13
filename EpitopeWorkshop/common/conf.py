import os

from EpitopeWorkshop.common import vars

DATA_HOME_DIR = os.getenv('DATA_DIR', './data')

DEFAULT_WINDOW_SIZE = 9
DEFAULT_WITH_SLIDING_WINDOW = True
DEFAULT_PARTITIONS_AMT = 8
DEFAULT_SS_PREDICTOR_THRESHOLD = 0.65
DEFAULT_NORMALIZE_VOLUME = False
DEFAULT_NORMALIZE_HYDROPHOBICITY = True
DEFAULT_NORMALIZE_SURFACE_ACCESSIBILITY = True
DEFAULT_TRAIN_DATA_PCT = 0.7
DEFAULT_TEST_DATA_PCT = 0.3
DEFAULT_VALID_DATA_PCT = 0.1
DEFAULT_EPOCHS = 3
DEFAULT_POS_WEIGHT = None
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_BATCH_SIZE = 20
DEFAULT_BATCHES_UNTIL_TEST = 20
DEFAULT_RECORDS_IN_FINAL_DF = 250000
DEFAULT_OVERSAMPLING_CHANGE_VAL_PROBA = 0.2
DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MIN = 90
DEFAULT_OVERSAMPLING_ALTERCATION_PCT_MAX = 110
DEFAULT_IS_IN_EPITOPE_THRESHOLD = 0.55
DEFAULT_PRESERVE_FILES_IN_PROCESS = True
DEFAULT_CONCURRENT_TRAIN_FILES_AMT = 25
NO_AMINO_ACID_CHAR = '-'
DEFAULT_USING_NET_DEVICE = 'cpu'
PATH_TO_USER_CNN_DIR = os.path.join(DATA_HOME_DIR, 'cnn-models')
PATH_TO_CNN_DIR = os.getenv('CNN_DIR', PATH_TO_USER_CNN_DIR)
CNN_NAME = 'cnn.pth'
USER_CNN_NAME = 'user-cnn.pth'
PATH_TO_CNN = os.path.join(PATH_TO_CNN_DIR, CNN_NAME)
PATH_TO_USER_CNN = os.path.join(DATA_HOME_DIR, USER_CNN_NAME)
HEAT_MAP_DIR = os.path.join(DATA_HOME_DIR, 'heat-maps')
PLOTS_DIR = os.path.join(DATA_HOME_DIR, 'plots')
DEFAULT_MIN_EPITOPE_SIZE = 4
DEFAULT_PRINT_PROBA = False
DEFAULT_PRINT_PRECISION = 3
DEFAULT_BALANCING_METHOD = vars.BALANCING_METHOD_UNDER

os.makedirs(PATH_TO_CNN_DIR, exist_ok=True)
os.makedirs(PATH_TO_USER_CNN_DIR, exist_ok=True)
os.makedirs(HEAT_MAP_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
