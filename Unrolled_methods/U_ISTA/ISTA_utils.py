import torch
import numpy as np


def shrink(u, tau, factor=1.):
    """Soft-thresholding of vector u at level tau > 0."""
    zer = torch.zeros_like(u)
    return torch.sign(u) * torch.maximum(zer, torch.abs(u) - tau)


def SNR(x_true, x_pred):
    """SNR of Batch"""
    n = 0
    d = 0
    sum_snr = 0
    for i in range(x_true.size()[0]):
        n = torch.linalg.norm(x_true[i])
        d = torch.linalg.norm((x_true-x_pred)[i])
        sum_snr = sum_snr+20*torch.log10(n/d)
    return (sum_snr/x_true.size()[0])


def TSNR(x_true, x_pred):
    """TSNR of one dimensional vector"""
    return 20*torch.log10(torch.linalg.norm(x_true[x_true != 0])/torch.linalg.norm(x_true[x_true != 0]-x_pred[x_true != 0]))
