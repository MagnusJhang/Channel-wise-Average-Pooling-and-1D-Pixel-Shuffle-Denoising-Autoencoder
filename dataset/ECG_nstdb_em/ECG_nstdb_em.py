import os
import json
import sys
sys.path.append("./dataset/ECG_nstdb_em/lib")
file_path = os.path.realpath(__file__)
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from lib.ParmDataset import *

class ECG_nstdb_em(Dataset):
    __memory_co = []
    __memory_gt = []

    NSTDB_snr_db = [0, 6, 12, 18, 24, -6, -1] # -1 represents select all noise
    DATASET_LEN = 650000

    def __init__(self, index_list, dump_memory=False):
        assert len(ECG_nstdb_em.__memory_co) != 0 \
            , "Run nstdb_database.allocateDataIndex(...) to obtain the data index first."
        self.__index = index_list
        self.dump_memory = dump_memory

    def __len__(self):
        return len(self.__index)

    def __getitem__(self, item):
        in_data = np.array(ECG_nstdb_em.__memory_co[self.__index[item]["index"]], copy=True)
        ou_data = np.array(ECG_nstdb_em.__memory_gt[self.__index[item]["index"]], copy=True)
        return [in_data,
                ou_data,
                {
                    "fname": self.__index[item]["fname"],
                    "ch": self.__index[item]["ch"],
                    "pos": self.__index[item]["pos"],
                    "min": self.__index[item]["min"],
                    "max": self.__index[item]["max"],
                    "snr_i": self.__index[item]["snr_i"]
                }
                ]

    @classmethod
    def allocate_data_index_nstdb(cls, params: ParmNSTDB):
        step = int(params.in_len * (1 - params.overlap_ratio))
        NOISE_LENGTH = 360 * 120 - step  # 2 min of noise segment
        NOISE_ST_INDEX = [108000, 194400, 280800, 367200, 453600, 540000, 626400]  # 5, 9, 13, 17, 21, 25, 29


        if params.execpt_db_file is not None:
            with open(params.execpt_db_file) as f:
                except_list = json.load(f)
        else:
            except_list = {}

        assert all(i in ECG_nstdb_em.NSTDB_snr_db for i in params.snr_db), \
            "Some snr level is not available in snr_db"

        if -1 in params.snr_db:
            if len(params.snr_db) > 1:
                raise Exception("You can not assign another snr level when -1 is declared in snr_db")
            else:
                params.snr_db = ECG_nstdb_em.NSTDB_snr_db[:-1]

        snr_str = [str(i).zfill(2) for i in params.snr_db]

        trainIndex = []
        testIndex = []

        file_list = os.listdir(params.db_dir)

        for fname in file_list:
            if fname.startswith("nstdb_") & fname.endswith(".mat"):
                code = fname.split("_")[1].split(".")[0]
                if any(i in code.split("e")[1] for i in snr_str):
                    snr_i = int(code.split("e")[1])

                    print("Loading {}".format(fname))
                    d = sio.loadmat("{}/{}".format(params.db_dir, fname))['data']
                    # cf = sio.loadmat("{}/{}".format(dir, cfname))['data']

                    # dshape = d['binary_zb'][0][0].shape
                    dshape = d['binary'][0][0].shape
                    dleads = [d['description'][0][0][0][0][0], d['description'][0][0][0][1][0]]

                    if params.zero_bias:
                        d_co = d['binary_zb'][0, 0].astype(float)
                        d_gt = d['binary_zb_clear'][0, 0].astype(float)
                    else:
                        d_co = d['binary'][0, 0].astype(float)
                        d_gt = d['binary_clear'][0, 0].astype(float)
                    st_idx = []

                    for i in NOISE_ST_INDEX:
                        st_idx = st_idx + np.arange(i, i + NOISE_LENGTH if (i + NOISE_LENGTH < dshape[0] - params.in_len)
                        else dshape[0] - params.in_len, step).tolist()

                    for i_ch in range(dshape[1]):
                        if params.leads[0].value != ECG_Leads.ALL.value and not (any(_x.name == dleads[i_ch] for _x in params.leads)):
                            continue
                        index = []
                        # add corrupted signal into input
                        current_key = "{}_{}".format(code.split("e")[0], i_ch)
                        valid_offset = int(len(st_idx) * (1 - params.valid_ratio))

                        for i in st_idx:
                            in_list = False
                            if current_key in except_list.keys():
                                for i_ex in except_list[current_key]['start']:
                                    if i_ex <= i <= i_ex + except_list['len'] \
                                            or i <= i_ex <= i + except_list['len']:
                                        in_list = True
                                        break
                                if in_list:
                                    continue

                            index.append(
                                {'index': len(ECG_nstdb_em.__memory_co), 'fname': fname[:-4], 'ch': i_ch, 'pos': i,
                                 'min': 0, 'max': 2 ** 11 - 1, "snr_i": snr_i})

                            if params.normalize.value == DataNorm.GLOBAL.value:
                                ECG_nstdb_em.__memory_co.append(
                                    np.divide(
                                        d_co[i:i + params.in_len, i_ch], 2 ** 11 - 1
                                    )
                                )
                                ECG_nstdb_em.__memory_gt.append(
                                    np.divide(
                                        d_gt[i:i + params.in_len, i_ch], 2 ** 11 - 1
                                    )
                                )

                            elif params.normalize.value == DataNorm.LOCAL.value:
                                index[-1]['min'] = np.min(d_co[i:i + params.in_len, i_ch])
                                ECG_nstdb_em.__memory_co.append(
                                    np.subtract(
                                        d_co[i:i + params.in_len, i_ch], index[-1]['min']
                                    )
                                )
                                ECG_nstdb_em.__memory_gt.append(
                                    np.subtract(
                                        d_gt[i:i + params.in_len, i_ch], index[-1]['min']
                                    )
                                )

                                index[-1]['max'] = np.max(ECG_nstdb_em.__memory_co[-1])
                                ECG_nstdb_em.__memory_co[-1] = np.divide(
                                    ECG_nstdb_em.__memory_co[-1], index[-1]['max']
                                )
                                ECG_nstdb_em.__memory_gt[-1] = np.divide(
                                    ECG_nstdb_em.__memory_gt[-1], index[-1]['max']
                                )
                            elif params.normalize.value == DataNorm.NONE.value:
                                ECG_nstdb_em.__memory_co = d_co[i:i + params.in_len, i_ch]
                                ECG_nstdb_em.__memory_gt = d_gt[i:i + params.in_len, i_ch]
                            else:
                                raise Exception("Can't recognize the normalize function {}".format(params.normalize))

                        train_limit = np.floor(len(index) * params.train_ratio).astype('int')
                        if params.random_idx:
                            rand_seed = np.random.choice(range(len(index)), len(index), replace=False).tolist()
                            for itr in range(train_limit):
                                trainIndex.append(index[rand_seed.pop()])
                            for _ in range(len(rand_seed)):
                                testIndex.append(index[rand_seed.pop()])
                        else:
                            trainIndex.extend(index[0:train_limit])
                            testIndex.extend(index[train_limit:len(index)])

        return [trainIndex, testIndex]


    @staticmethod
    def allocate_data_index(params_nstdb: ParmNSTDB = None):
        total_train_index = []
        total_test_index = []
        if params_nstdb is not None:
            train_index, test_index = ECG_nstdb_em.allocate_data_index_nstdb(params_nstdb)
            total_train_index.extend(train_index)
            total_test_index.extend(test_index)
        return [total_train_index, total_test_index]


    @staticmethod
    def LoadNstdbSegmentFromFile(nstdb_path, db_name, ch, pos, normalize: DataNorm, in_len=1024, zero_bias=False):
        d = sio.loadmat("{}/{}.mat".format(nstdb_path, db_name))['data']

        if zero_bias:
            d_co = d['binary_zb'][0, 0][pos:pos + in_len, ch].astype(float)
            d_gt = d['binary_zb_clear'][0, 0][pos:pos + in_len, ch].astype(float)
        else:
            d_co = d['binary'][0, 0][pos:pos + in_len, ch].astype(float)
            d_gt = d['binary_clear'][0, 0][pos:pos + in_len, ch].astype(float)

        if normalize.value == DataNorm.GLOBAL.value:
            d_co = np.divide(d_co, 2 ** 11 - 1)
            d_gt = np.divide(d_gt, 2 ** 11 - 1)

        elif normalize.value == DataNorm.LOCAL.value:
            min = np.min(d_co)
            d_co = np.subtract(d_co, min)
            d_gt = np.subtract(d_gt, min)
            max = np.max(d_co)
            d_co = np.divide(d_co, max)
            d_gt = np.divide(d_gt, max)

        return [d_co, d_gt]


if __name__ == "__main__":
    NSTDBPATH = "./raw"

    tr_index, te_index = ECG_nstdb_em.allocate_data_index(
        params_nstdb=ParmNSTDB(db_dir=NSTDBPATH, normalize=DataNorm.GLOBAL, random_idx=True)
    )

    tr_db = ECG_nstdb_em(tr_index)
    tr_loader = DataLoader(tr_db, batch_size=1, shuffle=True)
    print(tr_loader.__len__())

    plt.interactive(False)

    for i, [in_data, ou_data, attr] in enumerate(tr_loader, 1):
        in_data = torch.autograd.Variable(in_data.cuda().type(torch.float32))
        ou_data = torch.autograd.Variable(ou_data.cuda().type(torch.float32))
        plt.plot(in_data.cpu().numpy().reshape(-1))
        plt.plot(ou_data.cpu().numpy().reshape(-1))
        plt.title("{}[{}]_{}_snr:{}".format(attr["fname"][0], attr["ch"][0].cpu().item(), attr["pos"].item(),
                                            attr["snr_i"].item()))
        plt.show()
        plt.pause(10)