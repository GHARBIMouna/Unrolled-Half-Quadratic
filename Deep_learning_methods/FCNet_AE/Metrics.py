import torch


def SNR(x_true, x_pred):
    n = 0
    d = 0
    sum_snr = 0

    for i in range(x_true.size()[0]):
        n = torch.linalg.norm(x_true[i])
        d = torch.linalg.norm((x_true-x_pred)[i])
        sum_snr = sum_snr+20*torch.log10(n/d)

    return (sum_snr/x_true.size()[0])


def TSNR(x_true, x_pred):
    n = 0
    d = 0
    sum_tsnr = 0

    for i in range(x_true.size()[0]):
        n = torch.linalg.norm(x_true[x_true != 0][i])
        d = torch.linalg.norm((x_true[x_true != 0]-x_pred[x_true != 0])[i])
        sum_tsnr = sum_tsnr+20*torch.log10(n/d)

    return (sum_tsnr/x_true.size()[0])
