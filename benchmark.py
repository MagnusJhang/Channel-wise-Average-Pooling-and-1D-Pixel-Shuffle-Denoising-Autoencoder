import torch


def diff_pow(desire, pred):
    return (desire - pred) ** 2


def MSE(desire, pred):
    assert desire.shape == pred.shape
    return torch.sum(diff_pow(desire, pred), dim=2) / desire.shape[2]


def RMSE(desire, pred):
    assert desire.shape == pred.shape
    return MSE(desire, pred) ** 0.5


def PRD(desire, pred):
    assert desire.shape == pred.shape
    dd = torch.sum(diff_pow(desire, pred), dim=2)
    mm = torch.sum(desire ** 2, dim=2)
    return ((dd / mm) ** 0.5) * 100


def PRDN(desire, pred):
    assert desire.shape == pred.shape
    dd = torch.sum(diff_pow(desire, pred), dim=2)
    avg = torch.mean(desire, dim=2, keepdim=True)
    mm = torch.sum(diff_pow(desire, avg), dim=2)
    return ((dd / mm) ** 0.5) * 100


def SNR(signal, noise):
    assert signal.shape == noise.shape
    return torch.log10( torch.sum(signal**2, dim=2) / torch.sum(noise**2, dim=2)) * 10


def PSNR(desire, pred):
    assert desire.shape == pred.shape
    max = (torch.max(desire.view(desire.shape[0], -1), dim=1)[0])
    mse = MSE(desire, pred)
    return (max**2).view(-1, 1) / mse

def PSNR_I(desire, pred):
    assert desire.shape == pred.shape
    max = (torch.max(desire.view(desire.shape[0], -1), dim=1)[0])
    mse = MSE(desire, pred)
    return mse / (max**2).view(-1, 1)