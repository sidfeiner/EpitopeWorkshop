# Base fields
ID_COL_NAME = 'sequence_id'
SEQ_COL_NAME = 'sequence'
SUB_SEQ_INDEX_START_COL_NAME = 'sub_sequence_index_start'
SUB_SEQ_COL_NAME = 'sub_sequence'
AMINO_ACID_SEQ_INDEX_COL_NAME = 'amino_acid_seq_index'
AMINO_ACID_SUBSEQ_INDEX_COL_NAME = 'amino_acid_subseq_index'
IS_IN_EPITOPE_COL_NAME = 'is_in_epitope'

# Feature fields
CALCULATED_FEATURES = 'features'
ANALYZED_SEQ_COL_NAME = 'analyzed_sequence'
COMPUTED_VOLUME_COL_NAME = 'computed_volume'
HYDROPHOBICITY_COL_NAME = 'hydrophobicity'
RSA_COL_NAME = 'rsa'
SS_ALPHA_HELIX_PROBA_COL_NAME = 'secondary_structure_alpha_helix_proba'
SS_BETA_SHEET_PROBACOL_NAME = 'secondary_structure_beta_sheet_proba'
IS_POLAR_PROBA_COL_NAME = "is_polar_proba"

# Categorical type columns
IS_TYPE_A_COL_NAME = "is_type_A"
IS_TYPE_R_COL_NAME = "is_type_R"
IS_TYPE_N_COL_NAME = "is_type_N"
IS_TYPE_D_COL_NAME = "is_type_D"
IS_TYPE_C_COL_NAME = "is_type_C"
IS_TYPE_Q_COL_NAME = "is_type_Q"
IS_TYPE_E_COL_NAME = "is_type_E"
IS_TYPE_G_COL_NAME = "is_type_G"
IS_TYPE_H_COL_NAME = "is_type_H"
IS_TYPE_I_COL_NAME = "is_type_I"
IS_TYPE_L_COL_NAME = "is_type_L"
IS_TYPE_K_COL_NAME = "is_type_K"
IS_TYPE_M_COL_NAME = "is_type_M"
IS_TYPE_F_COL_NAME = "is_type_F"
IS_TYPE_P_COL_NAME = "is_type_P"
IS_TYPE_S_COL_NAME = "is_type_S"
IS_TYPE_T_COL_NAME = "is_type_T"
IS_TYPE_W_COL_NAME = "is_type_W"
IS_TYPE_Y_COL_NAME = "is_type_Y"
IS_TYPE_V_COL_NAME = "is_type_V"

TYPE_COLUMNS = {
    'A': IS_TYPE_A_COL_NAME,
    'R': IS_TYPE_R_COL_NAME,
    'N': IS_TYPE_N_COL_NAME,
    'D': IS_TYPE_D_COL_NAME,
    'C': IS_TYPE_C_COL_NAME,
    'Q': IS_TYPE_Q_COL_NAME,
    'E': IS_TYPE_E_COL_NAME,
    'G': IS_TYPE_G_COL_NAME,
    'H': IS_TYPE_H_COL_NAME,
    'I': IS_TYPE_I_COL_NAME,
    'L': IS_TYPE_L_COL_NAME,
    'K': IS_TYPE_K_COL_NAME,
    'M': IS_TYPE_M_COL_NAME,
    'F': IS_TYPE_F_COL_NAME,
    'P': IS_TYPE_P_COL_NAME,
    'S': IS_TYPE_S_COL_NAME,
    'T': IS_TYPE_T_COL_NAME,
    'W': IS_TYPE_W_COL_NAME,
    'Y': IS_TYPE_Y_COL_NAME,
    'V': IS_TYPE_V_COL_NAME,
}

FEATURES_ORDER = [
    COMPUTED_VOLUME_COL_NAME,
    HYDROPHOBICITY_COL_NAME,
    RSA_COL_NAME,
    SS_ALPHA_HELIX_PROBA_COL_NAME,
    SS_BETA_SHEET_PROBACOL_NAME,
    IS_POLAR_PROBA_COL_NAME,
]
FEATURES_ORDER.extend(TYPE_COLUMNS.values())

NETWORK_INPUT_ARGS = [COMPUTED_VOLUME_COL_NAME, HYDROPHOBICITY_COL_NAME, RSA_COL_NAME, SS_ALPHA_HELIX_PROBA_COL_NAME,
                      SS_BETA_SHEET_PROBACOL_NAME, IS_POLAR_PROBA_COL_NAME]
NETWORK_INPUT_ARGS.extend(TYPE_COLUMNS.values())

NETWORK_LABELED_INPUT_ARGS = NETWORK_INPUT_ARGS + [IS_IN_EPITOPE_COL_NAME]
