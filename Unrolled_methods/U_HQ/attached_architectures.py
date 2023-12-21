import torch.nn as nn
import torch
Soft = nn.Softplus()
# ReLU is the activation used during training, if local minima problems are encountered, try using Softplus.
r = nn.ReLU()


class R_Arch(torch.nn.Module):
    """"
    architectures to learn regularization parameter at each layer
    """""

    def __init__(self, Arch):
        super(R_Arch, self).__init__()

        self.architecture = Arch

        if self.architecture == 'lambda_Arch1':
            #U-HQ-FixN
            self.lamda_cvx = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

        if self.architecture == 'lambda_Arch2':
            self.fc_cvx = nn.Linear(2049, 1, bias=False)
            torch.nn.init.uniform_(self.fc_cvx.weight, a=0.001, b=0.002)

            self.fc_ncvx = nn.Linear(2049, 1, bias=False)
            torch.nn.init.uniform_(self.fc_ncvx.weight, a=0.001, b=0.002)

            self.gamma = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

        if self.architecture == 'lambda_Arch1_cvx':
            self.lamda_cvx = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

        if self.architecture == 'lambda_Arch1_ncvx':
            self.lamda_ncvx = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
        if self.architecture == 'lambda_Arch2_cvx':
            self.fc_cvx = nn.Linear(2049, 1, bias=False)
            torch.nn.init.uniform_(self.fc_cvx.weight, a=0.001, b=0.002)
            self.gamma = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
        if self.architecture == 'lambda_Arch2_ncvx':
            self.fc_ncvx = nn.Linear(2049, 1, bias=False)
            torch.nn.init.uniform_(self.fc_ncvx.weight, a=0.001, b=0.002)
            self.gamma = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
        if self.architecture == 'lamda_Arch1_cvx_overparam':
            self.lamda_cvx_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

        if self.architecture == 'lamda_Arch1_ncvx_overparam':
            self.lamda_ncvx_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

        if self.architecture == 'lamda_Arch1_overparam':
            #U-HQ-FixN-OverP
            self.lamda_cvx_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_cvx_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

            self.lamda_ncvx_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.lamda_ncvx_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

        if self.architecture == 'lamda_Arch2_overparam':
            #U-HQ
            self.fc_cvx = nn.Linear(2049, 1, bias=True)
            torch.nn.init.uniform_(self.fc_cvx.weight, a=0.001, b=0.002)

            self.fc_ncvx = nn.Linear(2049, 1, bias=True)
            torch.nn.init.uniform_(self.fc_ncvx.weight, a=0.001, b=0.002)

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

        if self.architecture == 'lamda_Arch2_cvx_overparam':
            self.fc_cvx = nn.Linear(2049, 1, bias=True)
            torch.nn.init.uniform_(self.fc_cvx.weight, a=0.001, b=0.002)

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

        if self.architecture == 'lamda_Arch2_ncvx_overparam':
            self.fc_ncvx = nn.Linear(2049, 1, bias=True)
            torch.nn.init.uniform_(self.fc_ncvx.weight, a=0.01, b=0.02)

            self.gamma_1 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_3 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_4 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_5 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_6 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_7 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_8 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_9 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)
            self.gamma_10 = nn.Parameter(
                torch.FloatTensor([1]).cuda(), requires_grad=True)

    def forward(self, H, x, xdeg):

        if self.architecture == 'lambda_Arch1':
            lamda_cvx = r(self.lamda_cvx)
            lamda_ncvx = r(self.lamda_ncvx)
            gamma = r(self.gamma)
            return (gamma, lamda_cvx, lamda_ncvx)

        if self.architecture == 'lambda_Arch2':
            res1 = ((torch.mm(H, x.T)-xdeg.T).T)**2
            lamda_cvx = r(self.fc_cvx(res1))
            lamda_ncvx = r(self.fc_ncvx(res1))
            gamma = r(self.gamma)
            return (gamma, lamda_cvx, lamda_ncvx)

        if self.architecture == 'lambda_Arch1_cvx':
            lamda_cvx = r(self.lamda_cvx)
            gamma = r(self.gamma)
            return gamma, lamda_cvx, torch.zeros_like(lamda_cvx)

        if self.architecture == 'lambda_Arch1_ncvx':
            lamda_ncvx = r(self.lamda_ncvx)
            gamma = r(self.gamma)
            return gamma, torch.zeros_like(lamda_ncvx), lamda_ncvx

        if self.architecture == 'lamda_Arch1_cvx_overparam':
            lamda_cvx = r(self.lamda_cvx_1*self.lamda_cvx_2*self.lamda_cvx_3*self.lamda_cvx_4*self.lamda_cvx_5 *
                          self.lamda_cvx_6*self.lamda_cvx_7*self.lamda_cvx_8*self.lamda_cvx_9*self.lamda_cvx_10)
            gamma = r(self.gamma_1*self.gamma_2*self.gamma_3*self.gamma_4*self.gamma_5 *
                      self.gamma_6*self.gamma_7*self.gamma_8*self.gamma_9*self.gamma_10)
            return gamma, lamda_cvx, torch.zeros_like(lamda_cvx)

        if self.architecture == 'lamda_Arch1_ncvx_overparam':
            lamda_ncvx = r(self.lamda_ncvx_1*self.lamda_ncvx_2*self.lamda_ncvx_3*self.lamda_ncvx_4*self.lamda_ncvx_5 *
                           self.lamda_ncvx_6*self.lamda_ncvx_7*self.lamda_ncvx_8*self.lamda_ncvx_9*self.lamda_ncvx_10)
            gamma = r(self.gamma_1*self.gamma_2*self.gamma_3*self.gamma_4*self.gamma_5 *
                      self.gamma_6*self.gamma_7*self.gamma_8*self.gamma_9*self.gamma_10)
            return gamma, torch.zeros_like(lamda_ncvx), lamda_ncvx

        if self.architecture == 'lamda_Arch1_overparam':
            lamda_cvx = r(self.lamda_cvx_1*self.lamda_cvx_2*self.lamda_cvx_3*self.lamda_cvx_4*self.lamda_cvx_5 *
                          self.lamda_cvx_6*self.lamda_cvx_7*self.lamda_cvx_8*self.lamda_cvx_9*self.lamda_cvx_10)
            lamda_ncvx = r(self.lamda_ncvx_1*self.lamda_ncvx_2*self.lamda_ncvx_3*self.lamda_ncvx_4*self.lamda_ncvx_5 *
                           self.lamda_ncvx_6*self.lamda_ncvx_7*self.lamda_ncvx_8*self.lamda_ncvx_9*self.lamda_ncvx_10)
            gamma = r(self.gamma_1*self.gamma_2*self.gamma_3*self.gamma_4*self.gamma_5 *
                      self.gamma_6*self.gamma_7*self.gamma_8*self.gamma_9*self.gamma_10)
            return gamma, lamda_cvx, lamda_ncvx

        if self.architecture == 'lambda_Arch2_cvx':
            res1 = ((torch.mm(H, x.T)-xdeg.T).T)**2

            lamda_cvx = r(self.fc_cvx(res1))
            gamma = r(self.gamma)
            return (gamma, lamda_cvx, torch.zeros_like(lamda_cvx))

        if self.architecture == 'lambda_Arch2_ncvx':

            res1 = ((torch.mm(H, x.T)-xdeg.T).T)**2
            lamda_ncvx = r(self.fc_ncvx(res1))
            gamma = r(self.gamma)
            return (gamma, torch.zeros_like(lamda_ncvx), lamda_ncvx)

        if self.architecture == 'lamda_Arch2_overparam':
            res1 = ((torch.bmm(H, x.unsqueeze(dim=2)) -
                    xdeg.unsqueeze(dim=2)).squeeze(dim=2))**2
            lamda_cvx = r(self.fc_cvx(res1))
            lamda_ncvx = r(self.fc_ncvx(res1))
            gamma = r(self.gamma_1*self.gamma_2*self.gamma_3*self.gamma_4*self.gamma_5 *
                      self.gamma_6*self.gamma_7*self.gamma_8*self.gamma_9*self.gamma_10)

            return (gamma, lamda_cvx, lamda_ncvx)

        if self.architecture == 'lamda_Arch2_cvx_overparam':
            res1 = ((torch.bmm(H, x.unsqueeze(dim=2)) -
                    xdeg.unsqueeze(dim=2)).squeeze(dim=2))**2
            lamda_cvx = r(self.fc_cvx(res1))
            gamma = r(self.gamma_1*self.gamma_2*self.gamma_3*self.gamma_4*self.gamma_5 *
                      self.gamma_6*self.gamma_7*self.gamma_8*self.gamma_9*self.gamma_10)
            return (gamma, lamda_cvx, torch.zeros_like(lamda_cvx))

        if self.architecture == 'lamda_Arch2_ncvx_overparam':
            res1 = ((torch.bmm(H, x.unsqueeze(dim=2)) -
                    xdeg.unsqueeze(dim=2)).squeeze(dim=2))**2
            lamda_ncvx = r(self.fc_ncvx(res1))
            gamma = r(self.gamma_1*self.gamma_2*self.gamma_3*self.gamma_4*self.gamma_5 *
                      self.gamma_6*self.gamma_7*self.gamma_8*self.gamma_9*self.gamma_10)
            return (gamma, torch.zeros_like(lamda_ncvx), lamda_ncvx)
