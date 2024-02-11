import torch.nn as nn
import torch
Soft = nn.Softplus()
R = nn.ReLU()


class stepsize_arch(torch.nn.Module):
    def __init__(self):
        super(stepsize_arch, self).__init__()
        self.tau = nn.Parameter(torch.DoubleTensor(
            [0.1]).cuda(), requires_grad=True)
        self.sigma = nn.Parameter(torch.DoubleTensor(
            [0.1]).cuda(), requires_grad=True)
        self.rho = nn.Parameter(torch.DoubleTensor(
            [0.1]).cuda(), requires_grad=True)

    def forward(self):
        tau = R(self.tau)
        sigma = R(self.sigma)
        rho = R(self.rho)
        return (tau, sigma, rho)
