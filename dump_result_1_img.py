import sys
import numpy as np

sys.path.append("./dataset/ECG_nstdb_em")
sys.path.append("./dataset/ECG_nstdb_em/lib")
sys.path.append("./dae_model")
sys.path.append("./dae_model/lib")

import torch
from matplotlib import pyplot as plt
import dae_model
from dataset.ECG_nstdb_em.lib.ParmDataset import *
from dataset.ECG_nstdb_em.ECG_nstdb_em import ECG_nstdb_em

if __name__ == "__main__":
    """parameter"""
    model_name = "CPDAE_Regular"
#    model_name = "CPDAE_Full"
#    model_name = "CPDAE_Lite"

    pwd_model_state_dict = "./output/bs32_{}/model_state_dict.pt".format(CPDAE_Regular)
    pwd_dataset = "./dataset/ECG_nstdb_em/raw"
    normailze = DataNorm.GLOBAL
    fname, ch, index, len = "nstdb_118e00", 0, 118000, 1024
    zero_bias = True

    model = getattr(dae_model, model_name)()
    model.load_state_dict(torch.load(pwd_model_state_dict))
    model.eval()

    [data_in, data_gt] = ECG_nstdb_em.LoadNstdbSegmentFromFile(
        nstdb_path=pwd_dataset,
        db_name=fname,
        ch=ch,
        pos=index,
        normalize=normailze,
        in_len=len,
        zero_bias=zero_bias
    )

    xaxis = np.linspace(index, index+len-1, len)
    tensor_in = torch.tensor(data_in).view(1, 1, -1).type(torch.float32)
    tensor_gt = torch.tensor(data_gt).view(1, 1, -1).type(torch.float32)
    tensor_pred = model(tensor_in)
    plt.figure()
    plt.plot(xaxis, tensor_in.cpu().detach().numpy().reshape(-1), label="Corrupted ECG")
    plt.plot(xaxis, tensor_gt.cpu().detach().numpy().reshape(-1), label="Clean ECG")
    plt.plot(xaxis, tensor_pred.cpu().detach().numpy().reshape(-1), label="Recnostructed ECG")
    plt.title("{}[{}][{}-{}]".format(fname, ch, index, index+len-1))
    plt.legend()
    plt.show()
    plt.pause(10)
