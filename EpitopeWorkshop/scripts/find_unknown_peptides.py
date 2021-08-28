import os
from glob import glob
from typing import Set, Optional

import fire
from Bio import SeqIO
from EpitopeWorkshop.process.read import load_sequences_from_fasta


class FindUnknownPeptides:
    def load_fasta_file(self, path: str) -> Set[str]:
        sequences = load_sequences_from_fasta(path)
        peptides_only = {item.lower() for item in sequences}
        return set(peptides_only)

    def find(self, known_peptides_fasta_file: str, unknown_structures_dir_path: str, limit: Optional[int] = None):
        known_peptides = self.load_fasta_file(known_peptides_fasta_file)
        pdb_files_wildcard = os.path.join(unknown_structures_dir_path, '*.pdb')
        pdb_files = glob(pdb_files_wildcard)
        for pdb_file_path in pdb_files:
            print(f"searching pdb file {pdb_file_path}")
            with open(pdb_file_path, 'r') as pdb_file:
                for record in SeqIO.parse(pdb_file, 'pdb-atom'):
                    if record.seq.lower() in known_peptides:
                        print(f"!!!!!!known chain found in {pdb_file_path}")
                    else:
                        print(f"unknown chain found in {pdb_file_path}: {record.seq.lower()}")


if __name__ == '__main__':
    fire.Fire(FindUnknownPeptides)
