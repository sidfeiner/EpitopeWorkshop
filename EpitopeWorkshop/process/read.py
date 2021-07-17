from Bio import SeqIO
from pandas import DataFrame

IDS_COL_NAME = 'id'
SEQ_COL_NAME = 'sequence'


def read_fasta_file(path: str) -> DataFrame:
    ids = []
    seqs = []
    with open(path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            ids.append(record.id)
            seqs.append(record.seq)
            if len(record.features):
                print(f"found features: {record.features}")

    return DataFrame(data={IDS_COL_NAME: ids, SEQ_COL_NAME: seqs}).set_index([IDS_COL_NAME])