import fire
from EpitopeWorkshop.process import read, features
from EpitopeWorkshop.common.conf import DEFAULT_WINDOW_SIZE, DEFAULT_WITH_SLIDING_WINDOW
from EpitopeWorkshop.process.features import FeatureCalculator


def main(sequences_file_path: str, with_sliding_window: bool = DEFAULT_WITH_SLIDING_WINDOW,
         window_size: int = DEFAULT_WINDOW_SIZE):
    df = read.read_fasta(sequences_file_path, with_sliding_window, window_size)
    calculator = FeatureCalculator()
    df = calculator.calculate_features(df)


if __name__ == '__main__':
    fire.Fire(main)
