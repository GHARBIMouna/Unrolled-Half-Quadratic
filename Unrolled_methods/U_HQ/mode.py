import torch
from tools import *


def MM(x, H, L, delta_s_cvx, delta_s_ncvx, lamda_cvx, lamda_ncvx, penalization_number_cvx, penalization_number_ncvx, mode):
    penal_name_cvx = 'omega_s' + str(penalization_number_cvx)
    penal_name_ncvx = 'omega_s' + str(penalization_number_ncvx)
    if mode == 'learning_lambda_MM' or mode == "Deep_equilibrium":
        Diag_cvx = torch.diag_embed(
            eval(penal_name_cvx + "(torch.matmul(L,x.T).T,delta_s_cvx)"))
        Diag_ncvx = torch.diag_embed(
            eval(penal_name_ncvx + "(torch.matmul(L,x.T).T,delta_s_ncvx)"))
        A = torch.inverse(torch.bmm(torch.transpose(H, 1, 2), H) + torch.mul(
            Diag_cvx, lamda_cvx.unsqueeze(1)) + torch.mul(Diag_ncvx, lamda_ncvx.unsqueeze(1)))
    return A
