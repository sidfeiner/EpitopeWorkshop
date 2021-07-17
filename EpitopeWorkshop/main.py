import fire
from EpitopeWorkshop.process import read, features


def main(pdb_file_path: str):
    df = read.read_fasta_file(pdb_file_path)
    df = features.calculate_features(df)


if __name__ == '__main__':
    fire.Fire(main)
