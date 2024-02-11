import torch.nn as nn
import torch
Soft = nn.Softplus()
R = nn.ReLU()


class stepsize_regularization_arch(torch.nn.Module):
    def __init__(self):
        super(stepsize_regularization_arch, self).__init__()
        self.gamma = nn.Parameter(torch.DoubleTensor(
            [0.2]).cuda(), requires_grad=True)
        self.xsi = nn.Parameter(torch.DoubleTensor(
            [4]).cuda(), requires_grad=True)

    def forward(self):
        gamma = R(self.gamma)
        xsi = R(self.xsi)
        return (gamma, xsi)
