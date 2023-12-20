import torch
import torch.nn as nn


"""Fair penalization"""


def phi_sfair(input, delta_s):
    y = delta_s * input / (abs(input) + delta_s)
    return y


def omega_sfair(u, delta_s):
    return delta_s / (abs(u) + delta_s)


def psi_sfair(input, delta):
    return delta * (torch.abs(input) - delta * torch.log(torch.abs(input) / delta + 1))


"""End Fair penalization"""

"""Tikhonov penalization"""


def phi_sTikhonov(input, delta_s):
    return (2*input)


def psi_sTikhonov(input, delta_s):
    return input**2


def omega_sTikhonov(input, delta_s):
    return 2 * torch.ones_like(input)


"""End Tikhonov penalization"""


"""Green Penlization"""


def psi_sgreen(input, delta):
    return torch.log(torch.cosh(input))


def phi_sgreen(input, delta):
    return torch.tanh(input)


def omega_sgreen(input, delta):
    return torch.tanh(input)/input


"""End Green penalization"""

"""Cauchy penalization"""


def phi_scauchy(input, delta_s):
    return (input*delta_s**2)/(delta_s**2+input**2)


def psi_scauchy(input, delta_s):
    return 0.5*delta_s**2*torch.log(1+(input**2)/delta_s**2)


def omega_scauchy(input, delta_s):
    return (delta_s**2)/(delta_s**2+input**2)


"""End Cauchy penalization"""


"""Welsh penalization"""


def psi_swelsh(input, delta):
    return (delta**2)*(1-torch.exp((-input ** 2) / (2 * delta ** 2)))


def phi_swelsh(input, delta):
    return (input) * torch.exp((-input ** 2) / (2 * delta ** 2))


def omega_swelsh(input, delta_s):
    return (torch.exp((-input ** 2) / (2 * delta_s**2)))


"""End Welsh penalization"""

"""Begin Geman MCClure"""


def phi_sGMc(input, delta):
    return (4 * (delta ** 4)) * input / (2 * (delta ** 2) + input ** 2) ** 2


def omega_sGMc(t, delta_s):
    return (4 * (delta_s ** 4)) / (2 * (delta_s ** 2) + t ** 2) ** 2


def psi_sGMc(input, delta):
    return (input ** 2)*(delta**2) / (2 * delta ** 2 + input ** 2)


"""end Geman MCClure"""


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
    return 20*torch.log10(torch.linalg.norm(x_true[x_true != 0])/torch.linalg.norm(x_true[x_true != 0]-x_pred[x_true != 0]))
