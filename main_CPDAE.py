#!/usr/bin/python3
"""system package"""
import ast
import json
import os
import argparse
import sys
from enum import Enum
"""pip package"""
import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from ptflops import get_model_complexity_info
"""project package"""
sys.path.append("./dataset/ECG_nstdb_em")
sys.path.append("./dataset/ECG_nstdb_em/lib")
sys.path.append("./dae_model")
sys.path.append("./dae_model/lib")
import benchmark
import dae_model
from dataset.ECG_nstdb_em.lib.ParmDataset import *
from dataset.ECG_nstdb_em.ECG_nstdb_em import ECG_nstdb_em


class DefaultArgument(Enum):
    dump_dir = "./output"
    nstdb_dir = "./dataset/ECG_nstdb_em/raw"
    batch_size = 32
    device = "cpu"
    epochs = 1000
    learning_rate = 2e-4
    optimizer = 'Adam'
    data_shuffle = True
    loss_func = 'MSE'
    dump_all_result_at_final = False


G_parser = argparse.ArgumentParser(description='The execution script for training/inference the [CPDAE_Lite / CPDAE_Regular / CPDAE_Full] model')
G_parser.add_argument('model_name', type=str,
                      help='Module name in dae_model(CPDAE_Lite, CPDAE_Full, or CPDAE_Regular)')

G_parser.add_argument('-dd', '--dump_dir', type=str, default=DefaultArgument.dump_dir.value,
                      help='The path to store the model result. (default: {})'.format(DefaultArgument.dump_dir.value))

G_parser.add_argument('-nd', '--nstdb_dir', type=str, default=DefaultArgument.nstdb_dir.value,
                      help='The path of nstdb dataset (default: {})'.format(DefaultArgument.nstdb_dir.value))

G_parser.add_argument('-bs', '--batch_size', type=int, metavar='N', default=DefaultArgument.batch_size.value,
                      help='batch size (default: {})'.format(DefaultArgument.batch_size.value))

G_parser.add_argument('-de', '--device', type=str, default=DefaultArgument.device.value,
                      help='The device type of tensor: [cuda, cuda:1, or cpu] (default: {})'.format(DefaultArgument.device.value))

G_parser.add_argument('-ep', '--epochs', type=int, default=DefaultArgument.epochs.value,
                      help='The number of epochs (default: {})'.format(DefaultArgument.epochs.value))

G_parser.add_argument('-lr', '--learning_rate', type=float, default=DefaultArgument.learning_rate.value,
                      help='Initial learning rate (default: {})'.format(DefaultArgument.learning_rate.value))

G_parser.add_argument('-op', '--optimizer', type=str, default=DefaultArgument.optimizer.value,
                      help='The usage optimizer: [Adam, SGD, etc] (default: {})'.format(DefaultArgument.optimizer.value))

G_parser.add_argument('-ds', '--data_shuffle', type=ast.literal_eval, default=DefaultArgument.data_shuffle.value,
                      help='Random the sequence of the index on training/testing set (default: {})'.format(DefaultArgument.data_shuffle.value))

G_parser.add_argument('-lf', '--loss_func', type=str, default=DefaultArgument.loss_func.value,
                      help="The function which would like used in backpropagation (default: {})".format(DefaultArgument.loss_func.value))

G_parser.add_argument('-df', '--dump_all_result_at_final', type=ast.literal_eval, default=DefaultArgument.dump_all_result_at_final.value,
                      help="Dump all the result of test dataset at final, it may take very long time. (default: {})".format(DefaultArgument.dump_all_result_at_final.value))
G_args = G_parser.parse_args()
del G_parser


"""Hyper-parameters"""
DUMP_IMG_PER_EPOCH = 50
BATCH_SIZE = G_args.batch_size
EPOCH = G_args.epochs
G_loss_func = getattr(benchmark, G_args.loss_func)
DE_NAME = G_args.device
DEVICE = torch.device(DE_NAME)
if DE_NAME.__contains__("cuda"):
    assert torch.cuda.is_available()
    try:
        torch.cuda.set_device(DEVICE.index)
    except:
        torch.cuda.set_device(0)
        DEVICE = torch.device("cuda:0")

"""Dataset"""
[G_trainIdx, G_testIdx] = ECG_nstdb_em.allocate_data_index(
    params_nstdb=ParmNSTDB(db_dir=G_args.nstdb_dir, normalize=DataNorm.GLOBAL, random_idx=True, zero_bias=True)
)

G_trDataset = ECG_nstdb_em(G_trainIdx)
G_teDataset = ECG_nstdb_em(G_testIdx)

"""Define Value"""
TQDM_BAR_PREFIX = "{l_bar}{bar}|{n_fmt}/{total_fmt} {elapsed}{postfix}]"
PLOT_LABEL = {"in": "IN: Corrupted ECG", "gt": "GT: Clean ECG", "ou": "OU: Reconstructed ECG"}


def export_result_to_img(d, fpwd, title="", xlabel=None, ylabel=None):
    fig = plt.figure()
    for i in d:
        plt.plot(i["value"], label=i["label"])
    plt.title(title)
    plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.draw()
    plt.savefig(fpwd)
    plt.close(fig)


def train(r_epochs, r_batch_size, r_model, r_optimizer, r_train_loader, r_record_folder, r_dump_img=False):
    r_model.train()
    r_optimizer.zero_grad()
    r_optimizer.step()

    pr_idx = 0
    prt_ba_idx = np.floor(np.linspace(0, len(r_train_loader) - 1, 6)[1:])

    train_loss = 0.0

    with tqdm(total=len(r_train_loader), bar_format=TQDM_BAR_PREFIX) as pbar:
        pbar.set_description("Epoch[{}]".format(r_epochs))
        for batch_idx, [in_data, ou_data, params] in enumerate(r_train_loader, 0):
            corrupted_ecg = torch.autograd.Variable(in_data.view(in_data.shape[0], 1, -1).to(DEVICE).type(torch.float32))
            clean_ecg = torch.autograd.Variable(ou_data.view(ou_data.shape[0], 1, -1).to(DEVICE).type(torch.float32))
            r_optimizer.zero_grad()
            reconstructed_ecg = r_model(corrupted_ecg)

            loss = torch.sum(G_loss_func(clean_ecg, reconstructed_ecg))
            train_loss += loss.cpu().detach().item()
            loss = loss / corrupted_ecg.shape[0]

            loss.backward()
            r_optimizer.step()

            if batch_idx in prt_ba_idx and r_dump_img:
                pr_idx += 1
                d = [{"value": np.squeeze(corrupted_ecg[0, :, :].cpu().detach().numpy()), "label": PLOT_LABEL["in"]},
                     {"value": np.squeeze(clean_ecg[0, :, :].cpu().detach().numpy()), "label": PLOT_LABEL["gt"]},
                     {"value": np.squeeze(reconstructed_ecg[0, :, :].cpu().detach().numpy()), "label": PLOT_LABEL["ou"]}]

                export_result_to_img(d,
                            fpwd='{}/{}/epoch{:04}_train{:02}.png'.format(r_record_folder, "dump", r_epochs, pr_idx),
                            title='train_epoch{},{}[{}][{}]\n{}'.format(r_epochs, params["fname"][0], params["ch"][0],
                                                                        params["pos"][0], r_model.get_model_net()))
            pbar.update(1)
            pbar.set_postfix_str(
                "train loss: {:.9f}".format(train_loss / (batch_idx * r_batch_size + corrupted_ecg.shape[0])))

    avg_tr_loss = train_loss / r_train_loader.dataset.__len__()

    return avg_tr_loss


def test(r_epochs, r_batch_size, r_model, r_test_loader, r_record_folder, r_dump_img=False):
    r_model.eval()
    test_loss = 0.0
    pic_cnt = 0

    with torch.no_grad():
        with tqdm(total=len(r_test_loader), bar_format=TQDM_BAR_PREFIX, colour="#55cc55") as pbar:
            pbar.set_description("Epoch[{}]".format(r_epochs))
            for batch_idx, [in_data, ou_data, params] in enumerate(r_test_loader, 0):
                corrupted_ecg = torch.autograd.Variable(in_data.view(in_data.shape[0], 1, -1).to(DEVICE).type(torch.float32))
                clean_ecg = torch.autograd.Variable(ou_data.view(ou_data.shape[0], 1, -1).to(DEVICE).type(torch.float32))
                reconstructed_ecg = r_model(corrupted_ecg)
                loss = torch.sum(G_loss_func(clean_ecg, reconstructed_ecg))

                test_loss += loss.cpu().detach().item()
                if pic_cnt < 10 and r_dump_img:
                    d = [{"value": np.squeeze(corrupted_ecg[0, :, :].cpu().detach().numpy()), "label": PLOT_LABEL["in"]},
                         {"value": np.squeeze(clean_ecg[0, :, :].cpu().detach().numpy()), "label": PLOT_LABEL["gt"]},
                         {"value": np.squeeze(reconstructed_ecg[0, :, :].cpu().detach().numpy()), "label": PLOT_LABEL["ou"]}]

                    export_result_to_img(d,
                                fpwd='{}/{}/epoch{:04}_test{:02}.png'.format(r_record_folder, "dump", r_epochs,
                                                                             pic_cnt),
                                title='test_epoch{},{}[{}][{}]\n{}'.format(r_epochs, params["fname"][0], params["ch"][0],
                                                                           params["pos"][0], r_model.get_model_net()))
                    pic_cnt += 1
                    del d

                pbar.set_postfix_str(
                    " test loss: {:.9f}".format(test_loss / (batch_idx * r_batch_size + corrupted_ecg.shape[0])))
                pbar.update(1)

        avg_test_loss = test_loss / r_test_loader.dataset.__len__()
    return avg_test_loss


def dump_all_criteria(r_model, r_loader, r_record_folder, r_dump_all_img):
    r_model.eval()

    SNRo_avg_dict = {}
    SNRimp_avg_dict = {}
    PRD_avg_dict = {}
    RMSE_avg_dict = {}

    pandas_data = pd.DataFrame(
        {"methods": [], "fname": [], "ch": [], "idx": [], "in_snr_dB": [], "ou_snr_dB": [], "SNR_imp": [], "PRD": [],
         "RMSE": []})

    in_snr_dB_arr = []

    with torch.no_grad():
        with tqdm(total=len(r_loader)) as pbar:
            pbar.set_description("Valid")
            for batch_idx, [in_data, ou_data, params] in enumerate(r_loader, 0):
                corrupted_ecg = torch.autograd.Variable(
                    in_data.view(in_data.shape[0], 1, -1).to(DEVICE).type(torch.float32))

                clean_ecg = torch.autograd.Variable(
                    ou_data.view(ou_data.shape[0], 1, -1).to(DEVICE).type(torch.float32))

                reconstructed_ecg = r_model(corrupted_ecg)
                snr_i = benchmark.SNR(clean_ecg, clean_ecg - corrupted_ecg)
                snr_o = benchmark.SNR(clean_ecg, clean_ecg - reconstructed_ecg)
                snr_imp = snr_o - snr_i
                prd = benchmark.PRD(clean_ecg, reconstructed_ecg)
                rmse = benchmark.RMSE(clean_ecg, reconstructed_ecg)

                record_snr_i = params["snr_i"].cpu().detach().item()
                if record_snr_i not in in_snr_dB_arr:
                    in_snr_dB_arr.append(record_snr_i)

                pandas_data = pandas_data.append({"methods": r_model.get_model_net(),
                                                  "fname": params["fname"][0],
                                                  "ch": params["ch"][0].cpu().detach().item(),
                                                  "idx": params["pos"][0].cpu().detach().item(),
                                                  "in_snr_dB": params["snr_i"][0].cpu().detach().item(),
                                                  "ou_snr_dB": snr_o.cpu().detach().item(),
                                                  "SNR_imp": snr_imp.cpu().detach().item(),
                                                  "PRD": prd.cpu().detach().item(),
                                                  "RMSE": rmse.cpu().detach().item()},
                                                 ignore_index=True)
                if r_dump_all_img:
                    export_result_to_img([
                        {"value": np.squeeze(corrupted_ecg[0, :, :].cpu().detach().numpy()), "label": PLOT_LABEL["in"]},
                        {"value": np.squeeze(clean_ecg[0, :, :].cpu().detach().numpy()), "label": PLOT_LABEL["gt"]},
                        {"value": np.squeeze(reconstructed_ecg[0, :, :].cpu().detach().numpy()), "label": PLOT_LABEL["ou"]}],
                                       fpwd='{}/test_{:04}.png'.format(r_record_folder, pic_cnt),
                                       title='{}[{}][{}]\n{}'.format(params["fname"][0], params["ch"][0],
                                                                 params["pos"][0], r_model.get_model_net()))
                    pic_cnt = pic_cnt + 1
                pbar.update(1)

    pandas_data.to_csv("{}/criteria_test.csv".format(r_record_folder), index=False)

    for snr_i in in_snr_dB_arr:
        filter_d = pandas_data.loc[pandas_data['in_snr_dB'] == snr_i]
        SNRimp_avg_dict[snr_i] = np.mean(filter_d['SNR_imp'])
        SNRo_avg_dict[snr_i] = np.mean(filter_d["ou_snr_dB"])
        PRD_avg_dict[snr_i] = np.mean(filter_d["PRD"])
        RMSE_avg_dict[snr_i] = np.mean(filter_d["RMSE"])
        print('SNRimp_avg[{} dB]: {:.9f}, PRD_avg[{} dB]: {:.9f}, RMSE_avg[{} dB]: {:.9f}'.format(
            snr_i, SNRimp_avg_dict[snr_i],
            snr_i, PRD_avg_dict[snr_i],
            snr_i, RMSE_avg_dict[snr_i]))

    return {"RMSE": RMSE_avg_dict, "SNRo": SNRo_avg_dict, "SNRimp": SNRimp_avg_dict, "PRD": PRD_avg_dict}


if __name__ == "__main__":
    train_loader = DataLoader(G_trDataset, batch_size=BATCH_SIZE, shuffle=G_args.data_shuffle)
    test_loader = DataLoader(G_teDataset, batch_size=BATCH_SIZE, shuffle=G_args.data_shuffle)

    if DEVICE.type == "cuda":
        model = getattr(dae_model, G_args.model_name)().cuda()
        sum = summary(model, (1, 1, 1024), verbose=0)
    else:
        model = getattr(dae_model, G_args.model_name)().cpu()
        sum = summary(model, (1, 1, 1024), verbose=0, device="cpu")

    optimizer = getattr(optim, G_args.optimizer)(model.parameters(), lr=G_args.learning_rate)

    scheduler_steplr = StepLR(optimizer, step_size=100, gamma=0.5)

    macs, params = get_model_complexity_info(model, (1, 1024), as_strings=False,
                                             print_per_layer_stat=False, verbose=False)
    print(sum.__repr__())

    target_relate_folder = 'bs{}_{}'.format(BATCH_SIZE, model.get_model_net())
    record_folder = '{}/{}'.format(G_args.dump_dir, target_relate_folder)
    if not os.path.exists(record_folder):
        os.makedirs("{}/dump".format(record_folder))
        os.makedirs("{}/test".format(record_folder))

    tr_loss_ep_list = []
    te_loss_ep_list = []

    dump_img_itr = np.arange(1, G_args.epochs + 1 - 10, DUMP_IMG_PER_EPOCH)
    # the interval of dump image which start from 1 and end to G_args.epochs + 1 - 10
    dump_img_itr = np.concatenate([dump_img_itr, np.arange(G_args.epochs - 9, G_args.epochs + 1, 1)])
    # dump img in last G_args.epochs + 1 - 10 to G_args.epochs

    print("Use {} to training and inference.".format(DE_NAME), end="  ")
    if DE_NAME.__contains__("cpu"):
        print("\033[1;35m If you want to use cuda device, please add the argument '-de cuda' when call "
              "main_CPDAE.py\033[0m")
    else:
        print("")

    for epochs in range(1, G_args.epochs + 1):
        tr_loss_ep = train(r_model=model,
                           r_record_folder=record_folder,
                           r_batch_size=BATCH_SIZE,
                           r_epochs=epochs,
                           r_optimizer=optimizer,
                           r_train_loader=train_loader,
                           r_dump_img=epochs in dump_img_itr)

        te_loss_ep = test(r_epochs=epochs,
                          r_model=model,
                          r_batch_size=BATCH_SIZE,
                          r_record_folder=record_folder,
                          r_test_loader=test_loader,
                          r_dump_img=epochs in dump_img_itr)

        tr_loss_ep_list.append(tr_loss_ep)
        te_loss_ep_list.append(te_loss_ep)
        scheduler_steplr.step()

    test_loader = DataLoader(G_teDataset, batch_size=1, shuffle=False)
    summary_log = dump_all_criteria(r_model=model, r_loader=test_loader,
                                r_record_folder="{}/test".format(record_folder),
                                r_dump_all_img=G_args.dump_all_result_at_final)

    torch.save(model.state_dict(), '{}/model_state_dict.pt'.format(record_folder))
    summary_dict = {
        'batch_size': BATCH_SIZE,
        "trainable_parmeter": sum.trainable_params,
        "macs": macs,
        "SNRimp": summary_log["SNRimp"],
        "PRD": summary_log["PRD"],
        "SNRo": summary_log["SNRo"],
        "RMSE": summary_log["RMSE"]
    }

    json.dump(summary_dict, open("./{}/summary.json".format(record_folder), "w"), indent=4, sort_keys=True)
    export_result_to_img([
        {"value": tr_loss_ep_list, "label": "Training loss"},
        {"value": te_loss_ep_list, "label": "Testing loss"}],
        fpwd="./{}/loss.png".format(record_folder),
        title="Model Loss per Epoch",
        xlabel="Number of Epoch",
        ylabel="Mean Square Error")

    pandas_data = pd.DataFrame(
        {"training loss": tr_loss_ep_list, "testing loss": te_loss_ep_list})
    pandas_data.to_csv("{}/loss.csv".format(record_folder), index=False)
    os.sync()
    torch.cuda.empty_cache()