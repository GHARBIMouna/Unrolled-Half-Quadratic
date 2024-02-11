from PD_utils import activation_primal, activation_dual
from stepsize_architecture import stepsize_arch
import torch
import torch.nn as nn


class layer(torch.nn.Module):
    def __init__(self):
        super(layer, self).__init__()
        self.architecture = stepsize_arch()

    def forward(self, H, p, p_old, d, d_old, y):
        tau, sigma, rho = self.architecture()
        bp = -tau*torch.bmm(torch.transpose(H, 1, 2),
                            torch.unsqueeze(2*d-d_old, 2)).squeeze(2)
        p = activation_primal(p+bp, tau)
        bd = sigma*torch.bmm(H, torch.unsqueeze(p, 2)).squeeze(2)
        d = activation_dual(d+bd, y, sigma, rho)
        return p, d


class PD_model(torch.nn.Module):
    def __init__(self, num_layers):
        super(PD_model, self).__init__()
        self.Layers = nn.ModuleList()
        for i in range(num_layers):
            self.Layers.append(layer())

    def forward(self, H, p0, p0_old, d0, d0_old, y):
        for i, l in enumerate(self.Layers):
            p, d = self.Layers[i](H, p0, p0_old, d0, d0_old, y)
            p0_old = p0
            d0_old = d0
            d0 = d
            p0 = p
        return p, d
