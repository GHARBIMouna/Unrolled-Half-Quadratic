from ISTA_utils import shrink
import torch
import torch.nn as nn
from ISTA_joint_architecture import stepsize_regularization_arch


class layer(torch.nn.Module):
    def __init__(self):
        super(layer, self).__init__()
        self.architecture = stepsize_regularization_arch()

    def forward(self, H, x, y):
        gamma, xsi = self.architecture()
        x = shrink(x-gamma*torch.squeeze(torch.bmm(torch.transpose(H, 1, 2),
                   (torch.bmm(H, torch.unsqueeze(x, 2)) - torch.unsqueeze(y, 2))), 2), gamma*xsi)
        return x


class ISTA_model(torch.nn.Module):
    def __init__(self, num_layers):
        super(ISTA_model, self).__init__()
        self.Layers = nn.ModuleList()
        for i in range(num_layers):
            self.Layers.append(layer())

    def forward(self, H, x0, y, x_true):
        for i, l in enumerate(self.Layers):
            x = self.Layers[i](H, x0, y)
            x0 = x
        return x
