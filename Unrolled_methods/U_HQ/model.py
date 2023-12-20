import torch.nn as nn
import torch
from mode import  *
from attached_architectures import R_Arch
from tools import *
r = nn.ReLU()
Soft = nn.Softplus()


class Iter(torch.nn.Module):
    def __init__(self, mode, architecture_lambda):
        super(Iter, self).__init__()
        self.mode = mode
        self.architecture_name = architecture_lambda
        if mode == "learning_lambda_MM":
            self.architecture = R_Arch(architecture_lambda)

    def forward(self, x, xdeg, Ht_x_degraded, xtrue, H, L, delta_s_cvx,
                delta_s_ncvx, penalization_number_cvx,
                penalization_number_ncvx, Disp_param, lamda_cvx=None, lamda_ncvx=None):

        penal_name_cvx = 'phi_s' + str(penalization_number_cvx)
        penal_name_ncvx = 'phi_s' + str(penalization_number_ncvx)

        if self.mode == "learning_lambda_MM":
            gamma, lamda_cvx, lamda_ncvx = self.architecture(H, x, xdeg)

            first_branch = torch.bmm(torch.bmm(torch.transpose(
                H, 1, 2), H), x.unsqueeze(dim=2)) - Ht_x_degraded

            second_branch = eval(
                penal_name_cvx + "(x.unsqueeze(dim=2),delta_s_cvx)")
            second_branch1 = eval(
                penal_name_ncvx + "(x.unsqueeze(dim=2),delta_s_ncvx)")

            if self.architecture_name in ["lambda_Arch1", "lambda_Arch1_cvx", "lambda_Arch1_ncvx", "lamda_Arch1_overparam", "lamda_Arch1_cvx_overparam", "lamda_Arch1_ncvx_overparam"]:
                summ = lamda_cvx*second_branch + first_branch + lamda_ncvx*second_branch1
            else:
                summ = lamda_cvx.unsqueeze(
                    dim=2) * second_branch + first_branch + lamda_ncvx.unsqueeze(dim=2)*second_branch1

            inv_A = MM(x, H, L, delta_s_cvx, delta_s_ncvx, lamda_cvx,
                       lamda_ncvx, penalization_number_cvx, penalization_number_ncvx, self.mode)
            x = (x.unsqueeze(dim=2) - gamma *
                 torch.bmm(inv_A, summ)).squeeze(dim=2)

        if self.mode == "Deep_equilibrium":

            first_branch = torch.bmm(torch.bmm(torch.transpose(
                H, 1, 2), H), x.unsqueeze(dim=2)) - Ht_x_degraded

            second_branch = eval(
                penal_name_cvx + "(x.unsqueeze(dim=2),delta_s_cvx)")
            second_branch1 = eval(
                penal_name_ncvx + "(x.unsqueeze(dim=2),delta_s_ncvx)")
            summ = lamda_cvx * second_branch + first_branch + lamda_ncvx*second_branch1
            inv_A = MM(x,  H, L, delta_s_cvx, delta_s_ncvx, lamda_cvx,
                       lamda_ncvx, penalization_number_cvx, penalization_number_ncvx, self.mode)
            x = (x.unsqueeze(dim=2) - torch.bmm(inv_A, summ)).squeeze(dim=2)

        if Disp_param == True:
            return x, lamda_cvx, lamda_ncvx, gamma

        if Disp_param == False:
            return x


class Block(torch.nn.Module):

    def __init__(self,  mode, architecture_lambda):
        super(Block, self).__init__()

        self.Iter = Iter(mode, architecture_lambda)

    def forward(self, x, xdeg, Ht_x_degraded, xtrue, H, L, delta_s, delta_s1, penalization_num,
                penalization_num1, Disp_param, lamda_cvx=None, lamda_ncvx=None):

        return self.Iter(x, xdeg, Ht_x_degraded, xtrue, H, L, delta_s, delta_s1, penalization_num, penalization_num1, Disp_param, lamda_cvx, lamda_ncvx)


class myModel(torch.nn.Module):

    def __init__(self, number_layers, L, delta_s_cvx, delta_s_ncvx, mode, number_penalization_cvx,
                 number_penalization_ncvx, architecture_lambda):
        super(myModel, self).__init__()

        self.Layers = nn.ModuleList()
        self.L = L
        self.delta_s_cvx = delta_s_cvx
        self.delta_s_ncvx = delta_s_ncvx
        self.mode = mode
        self.number_penalization_cvx = number_penalization_cvx
        self.number_penalization_ncvx = number_penalization_ncvx
        self.architecture_lambda = architecture_lambda

        for i in range(number_layers):
            self.Layers.append(Block(mode, architecture_lambda))

        if mode == "Deep_equilibrium" or mode == "Deep_equilibrium_3MG":
            self.architecture = R_Arch(architecture_lambda)

    def forward(self, x, xdeg, xtrue, Ht_x_degraded, H, mode, Disp_param):
        if Disp_param == True:
            lambdas_cvx = []
            lambdas_ncvx = []
            gammas = []

        for i, l in enumerate(self.Layers):

            if (mode == "learning_lambda_MM"):

                if Disp_param == False:
                    x = self.Layers[i](x, xdeg, Ht_x_degraded, xtrue, H, self.L, self.delta_s_cvx, self.delta_s_ncvx,
                                       self.number_penalization_cvx, self.number_penalization_ncvx, Disp_param)
                if Disp_param == True:
                    x, lambda_cvx, lambda_ncvx, gamma = self.Layers[i](x, xdeg, Ht_x_degraded, xtrue, H, self.L,
                                                                       self.delta_s_cvx, self.delta_s_ncvx,
                                                                       self.number_penalization_cvx, self.number_penalization_ncvx, Disp_param)

                    lambdas_cvx.append(lambda_cvx)
                    lambdas_ncvx.append(lambda_ncvx)
                    gammas.append(gamma)

            if (mode == 'Deep_equilibrium'):
                lamda_cvx, lamda_ncvx = self.architecture(H, x, xdeg)
                x = self.Layers[i](x, xdeg, Ht_x_degraded, xtrue, H, self.L, self.delta_s_cvx, self.delta_s_ncvx, self.number_penalization_cvx,
                                   self.number_penalization_ncvx, Disp_param, lamda_cvx, lamda_ncvx)

        if Disp_param == False:
            return r(x)
        if Disp_param == True:
            return r(x), lambdas_cvx, lambdas_ncvx, gammas
