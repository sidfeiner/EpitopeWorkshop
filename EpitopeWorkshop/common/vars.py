from Bio.SeqUtils import ProtParamData

AMINO_ACIDS = ProtParamData.kd.keys()

# Polarity based on: https://teaching.ncl.ac.uk/bms/wiki/index.php/Amino_acids
POLAR_AMINO_ACIDS = ['R', 'N', 'D', 'C', 'Q', 'E', 'H', 'K', 'S', 'T', 'Y']
AMINO_ACIDS_POLARITY_MAPPING = {aa: aa in POLAR_AMINO_ACIDS for aa in AMINO_ACIDS}

BALANCING_METHOD_UNDER = 'under'
BALANCING_METHOD_OVER = 'over'
BALANCING_METHOD_NONE = 'none'
