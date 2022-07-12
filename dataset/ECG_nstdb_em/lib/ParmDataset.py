import json
from enum import Enum
 
__all__ = [
    'ECG_Leads',
    'DataNorm',
    'NoiseType',
    'ParmNSTDB'
]

class ECG_Leads(Enum):
    ALL = -1
    MLII = 0
    V1 = 1
    V2 = 2
    V4 = 3
    V5 = 4


class DataNorm(Enum):
    GLOBAL = 0
    LOCAL = 1
    NONE = 2


class NoiseType(Enum):
    BW = "bw"
    EM = "em"
    MA = "ma"


class ParmNSTDB:
    def __init__(self, db_dir,
                 normalize: DataNorm,
                 in_len=1024,
                 leads: list = None,
                 snr_db: list = None,
                 valid_ratio=0.0,
                 train_ratio=0.8,
                 overlap_ratio=0.0,
                 random_idx=True,
                 zero_bias=False,
                 execpt_db_file: str = None):

        self.leads = [ECG_Leads.ALL] if (leads is None) else leads
        self.snr_db = [-1] if (snr_db is None) else snr_db

        self.db_dir = db_dir
        self.in_len = in_len

        self.train_ratio = train_ratio
        self.overlap_ratio = overlap_ratio
        self.random_idx = random_idx
        self.zero_bias = zero_bias
        self.valid_ratio = valid_ratio
        self.normalize = normalize
        self.execpt_db_file = execpt_db_file