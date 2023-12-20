import torch
import time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from tools import SNR, TSNR
from model import myModel
from tools import *
import torch.nn.functional as FF
from modules import MyDataset


class U_HQ_class(nn.Module):
    def __init__(self, initializations, train_conditions, test_conditions, number_penalization, number_penalization1, folders, mode):
        super(U_HQ_class, self).__init__()
        self.initial_x0, self.number_layers, self.L, self.delta_s, self.delta_s1 = initializations

        self.number_epochs, self.train_batch_size, self.val_batch_size, self.architecture_lambda = train_conditions
        self.test_batch_size = test_conditions[0]
        self.path_train, self.path_validation, self.path_test, self.path_save = folders

        self.mode = mode
        self.number_penalization = number_penalization
        self.number_penalization1 = number_penalization1
        self.dtype = torch.cuda.FloatTensor

        self.model = myModel(self.number_layers, self.L, self.delta_s,
                             self.delta_s1, self.mode, self.number_penalization,
                             self.number_penalization1, self.architecture_lambda).cuda()

        self.loss_fun = nn.MSELoss()

    def CreateLoader(self, need_names, path_set=None):

        if path_set is not None:
            without_extra = os.path.normpath(path_set)
            last_part = os.path.basename(without_extra)
            if last_part == "training":
                train_data = MyDataset(
                    self.path_train, self.initial_x0, need_names)
                self.loader = DataLoader(
                    train_data, batch_size=1, shuffle=False)
            if last_part == "validation":
                val_data = MyDataset(self.path_validation,
                                     self.initial_x0, need_names)
                self.loader = DataLoader(val_data, batch_size=1, shuffle=False)
            if last_part == "test":
                test_data = MyDataset(
                    self.path_test, self.initial_x0, need_names)
                self.loader = DataLoader(
                    test_data, batch_size=1, shuffle=False)
            self.size = len(self.loader)
        else:

            # For training purposes
            train_data = MyDataset(
                self.path_train, self.initial_x0, need_names)
            self.train_loader = DataLoader(
                train_data, batch_size=self.train_batch_size, shuffle=True)
            val_data = MyDataset(self.path_validation,
                                 self.initial_x0, need_names)
            self.val_loader = DataLoader(
                val_data, batch_size=self.val_batch_size, shuffle=True)

    def train(self, lr, number_training, need_names="no", path_model=None):

        if self.mode == "learning_lambda_MM" or self.mode == "Deep_equilibrium":
            if self.architecture_lambda in ["lamda_Arch2_overparam", "lamda_Arch2_cvx_overparam", "lamda_Arch2_ncvx_overparam"]:
                my_list = []
                lr_list = 1e-2
                overparam_N = 10
                for i in range(self.number_layers):
                    for j in range(1, overparam_N+1):
                        my_list.append("Layers."+str(i) +
                                       ".Iter.architecture.gamma_"+str(j))

                params = list(map(lambda x: x[1], list(
                    filter(lambda kv: kv[0] in my_list, self.model.named_parameters()))))
                base_params = list(map(lambda x: x[1], list(
                    filter(lambda kv: kv[0] not in my_list, self.model.named_parameters()))))
                optimizer = optim.Adam(
                    [{'params': base_params}, {'params': params, 'lr': lr_list}], lr=lr)

            elif self.architecture_lambda in ["lambda_Arch1", "lambda_Arch1_cvx", "lambda_Arch1_ncvx", "lamda_Arch1_overparam", "lamda_Arch1_cvx_overparam", "lamda_Arch1_ncvx_overparam"]:
                optimizer = optim.Adam(self.model.parameters(), lr=lr)

            epoc = 0
            if path_model is not None:
                checkpoint = torch.load(path_model)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                epoc = checkpoint['epoch'] + 1

            else:
                if not os.path.exists(os.path.join(self.path_save, 'Trained_Model_'+self.mode+'_'+str(number_training))):
                    os.makedirs(os.path.join(
                        self.path_save, 'Trained_Model_'+self.mode+'_'+str(number_training)))
                file_object = open(os.path.join(
                    self.path_save, 'Trained_Model_'+self.mode+'_'+str(number_training)) + "/readme.txt", "a")
                file_object.writelines(["Mode: " + self.mode + '\n',
                                        "Optimizer: " + str(optimizer) + '\n',
                                        "learning_rate:" + str(lr) + '\n',
                                        "Number layers: " +
                                        str(self.number_layers) + '\n',
                                        "Penalization_cvx:" +
                                        str(self.number_penalization) + '\n',
                                        "Penalization_ncvx:" +
                                        str(self.number_penalization1) + '\n',
                                        "Delta_cvx: " +
                                        str(self.delta_s) + '\n',
                                        "Delta_ncvx: " +
                                        str(self.delta_s1) + '\n',
                                        "Initial_x0: " +
                                        str(self.initial_x0) + '\n',
                                        "batch_train_size: " +
                                        str(self.train_batch_size) + '\n',
                                        "batch_val_size: " +
                                        str(self.val_batch_size) + '\n',
                                        "Architecture_lambda: " +
                                        str(self.architecture_lambda) + '\n',
                                        ])
                file_object.close()

            self.CreateLoader(need_names=need_names)
            loss_epochs = []
            val_loss_epochs = []
            for epoch in range(self.number_epochs):
                print("Epoch ", epoch)
                if epoch+epoc == 0:
                    self.model.Layers.eval()
                if epoch+epoc > 0:
                    self.model.Layers.train()
                running_loss = 0.0
                total_SNR = 0
                for i, minibatch in enumerate(self.train_loader, 0):
                    if self.initial_x0 == "Null_initialization":
                        if need_names == "yes":
                            [name, H, x_true, x_degraded] = minibatch

                        if need_names == "no":
                            [H, x_true, x_degraded] = minibatch

                        x_true = Variable((x_true).type(
                            self.dtype), requires_grad=False)
                        H = Variable((H).type(self.dtype), requires_grad=False)
                        x_degraded = Variable((x_degraded).type(
                            self.dtype), requires_grad=False)
                        x0 = Variable(torch.zeros((self.train_batch_size, x_true.size()[1])).type(
                            self.dtype), requires_grad=False)

                    Ht_x_degraded = torch.bmm(torch.transpose(
                        H, 1, 2), x_degraded.unsqueeze(dim=2)).detach()

                    if epoch+epoc == 0:
                        x_pred = self.model(
                            x0, x_degraded, x_true, Ht_x_degraded, H, self.mode, False).detach()
                        loss = self.loss_fun(x_pred, x_true)
                        snr = SNR(x_true, x_pred).detach()
                        running_loss += loss.item()
                        total_SNR += snr
                    if epoch+epoc > 0:
                        optimizer.zero_grad()
                        x_pred = self.model(
                            x0, x_degraded, x_true, Ht_x_degraded, H, self.mode, False)
                        loss = self.loss_fun(x_pred, x_true)
                        loss.backward()
                        optimizer.step()
                        snr = SNR(x_true, x_pred).detach()
                        running_loss += loss.item()
                        total_SNR += snr
                        torch.autograd.set_detect_anomaly(True)
                loss_epochs.append(running_loss / (len(self.train_loader)))
                print("Train AVG MSE for Epoch ", epoch + epoc, "is",
                      float(running_loss / (len(self.train_loader))))
                print("Train AVG SNR for Epoch ", epoch + epoc, "is",
                      float(total_SNR/(len(self.train_loader))))

                # saving model here
                torch.save({
                    'epoch': epoch + epoc,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss, },
                    os.path.join(self.path_save, 'Trained_Model_'+self.mode+'_'+str(number_training)) + '/epoch' + str(epoch + epoc))

                # Evaluation on validation set
                self.model.eval()
                with torch.no_grad():
                    loss_current_val = 0
                    total_SNR_val = 0
                    for minibatch in self.val_loader:
                        if self.initial_x0 == "Null_initialization":
                            if need_names == "yes":
                                [name, H, x_true, x_degraded] = minibatch
                            if need_names == "no":
                                [H, x_true, x_degraded] = minibatch

                            x_true = Variable((x_true).type(
                                self.dtype), requires_grad=False)
                            x_degraded = Variable((x_degraded).type(
                                self.dtype), requires_grad=False)
                            H = Variable((H).type(self.dtype),
                                         requires_grad=False)
                            x0 = Variable(torch.zeros((self.val_batch_size, x_true.size()[1])).type(
                                self.dtype), requires_grad=False)

                        Ht_x_degraded = torch.bmm(torch.transpose(
                            H, 1, 2), x_degraded.unsqueeze(dim=2)).detach()
                        x_pred = self.model(
                            x0, x_degraded, x_true, Ht_x_degraded, H, self.mode, False)
                        loss_val = self.loss_fun(x_pred, x_true)
                        snr_val = SNR(x_true, x_pred)
                        loss_current_val += torch.Tensor.item(loss_val)
                        total_SNR_val += snr_val
                    val_loss_epochs.append(
                        loss_current_val / (len(self.val_loader)))
                    print("Val AVG MSE for Epoch ", epoch + epoc, "is",
                          float(loss_current_val / (len(self.val_loader))))
                    print("Val AVG SNR for Epoch ", epoch + epoc, "is",
                          float(total_SNR_val/(len(self.val_loader))))
                    # saving learning evolution in readme file
                    file_object = open(
                        os.path.join(self.path_save, 'Trained_Model_' +
                                     self.mode + '_' + str(number_training)) + "/readme.txt",
                        "a")
                    file_object.writelines(["Train loss for epoch " + str(epoch + epoc) + "is: " + str(running_loss / (len(self.train_loader)))+'\n', "Train SNR for epoch " + str(epoch + epoc) + "is: " + str(total_SNR / (len(self.train_loader)))+'\n',
                                            "Val loss for epoch" + str(epoch + epoc) + "is: " + str(loss_current_val / (len(self.val_loader))) + '\n', "Val SNR for epoch" + str(
                        epoch + epoc) + "is: " + str(total_SNR_val / (len(self.val_loader))) + '\n'
                    ])
                    file_object.close()
                    # plotting learning curves
                    plt.plot(loss_epochs, color='black',
                             linestyle='dashed', linewidth=1)
                    plt.plot(val_loss_epochs, color='blue',
                             linestyle='dashed', linewidth=1)
                    axes = plt.gca()
                    axes.set_xlabel('Epochs')
                    axes.set_ylabel('Average loss')
                    plt.savefig(os.path.join(os.path.join(self.path_save, 'Trained_Model_' + self.mode + '_' + str(
                        number_training)), 'Trained_Model_' + self.mode + '_' + str(number_training)+"_Loss_curve.png"))
            return

    def test(self, path_set=None, path_model=None, need_names="no", path_signal=None, save_estimate=False, Disp_param=False):

        if self.mode == "learning_lambda_MM" or self.mode == 'Deep_equilibrium':
            checkpoint = torch.load(path_model, map_location='cuda:0')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

        if path_signal is not None:

            name = os.path.split(path_signal)[1]
            x_true = torch.unsqueeze(torch.tensor(
                np.load(path_signal, allow_pickle=True)), 0)
            x_degraded = torch.unsqueeze(torch.tensor(np.load(path_signal.replace(
                'Groundtruth', 'Degraded').replace('_Gr_', '_De_'), allow_pickle=True)), 0)

            x_true = Variable(x_true.type(self.dtype), requires_grad=False)
            x_degraded = Variable((x_degraded).type(
                self.dtype), requires_grad=False)
            H = torch.unsqueeze(torch.tensor(np.load(path_signal.replace(
                'Groundtruth', 'H').replace('x_Gr', 'H'), allow_pickle=True)), 0)

            H = Variable((H).type(self.dtype), requires_grad=False)

            if self.initial_x0 == "Null_initialization":
                x_0 = Variable(torch.zeros((1, x_true.size()[1])).type(
                    self.dtype), requires_grad=False)
            Ht_x_degraded = torch.bmm(torch.transpose(
                H, 1, 2), x_degraded.unsqueeze(dim=2)).detach()
            t0 = time.time()
            x_pred = self.model(x_0, x_degraded, x_true,
                                Ht_x_degraded, H, self.mode, False).detach()
            t1 = time.time()
            loss = self.loss_fun(x_pred, x_true).detach()
            snr = SNR(x_true, x_pred)
            tsnr = TSNR(x_true, x_pred)
            print("loss is:", loss, "and SNR is", snr,  "and TSNR is", tsnr)
            print("Execution time is", t1-t0)
            return x_pred

        else:
            with torch.no_grad():
                self.CreateLoader(need_names, path_set)
                total_loss = 0
                total_time = 0
                high_MSE = 0
                total_SNR = 0
                total_tSNR = 0
                i = 0
                MSE_list = []
                SNR_list = []
                TSNR_list = []
                for minibatch in self.loader:
                    if self.initial_x0 == "Null_initialization":

                        if need_names == "yes":
                            name, H, x_true, x_degraded = minibatch

                        if need_names == "no":
                            H, x_true, x_degraded = minibatch
                        x_true = Variable(x_true.type(
                            self.dtype), requires_grad=False)
                        x_degraded = Variable((x_degraded).type(
                            self.dtype), requires_grad=False)
                        H = Variable(H.type(self.dtype), requires_grad=False)
                        x_0 = Variable(torch.zeros((self.test_batch_size, x_true.size()[1])).type(
                            self.dtype), requires_grad=False)

                    Ht_x_degraded = torch.bmm(torch.transpose(
                        H, 1, 2), x_degraded.unsqueeze(dim=2)).detach()
                    if Disp_param == True:
                        x_pred, lambdas_cvx, lambdas_ncvx, gammas = self.model(
                            x_0, x_degraded, x_true, Ht_x_degraded, H, self.mode, True)

                    t0 = time.time()
                    x_pred = self.model(
                        x_0, x_degraded, x_true, Ht_x_degraded, H, self.mode, False)
                    t1 = time.time()
                    loss = (self.loss_fun(x_true, x_pred).detach())
                    snr = SNR(x_true, x_pred).detach()
                    tsnr = TSNR(x_true, x_pred).detach()
                    total_loss += loss
                    total_time += t1-t0
                    total_SNR += snr
                    total_tSNR += tsnr

                    MSE_list.append(loss)
                    SNR_list.append(snr)
                    TSNR_list.append(tsnr)
                    i += 1
                # compute metrics STD
                mse_std = 0
                for l in MSE_list:
                    mse_std = mse_std+((l-total_loss/self.size)**2)
                mse_std = torch.sqrt(mse_std/(self.size-1))

                snr_std = 0
                for l in SNR_list:
                    snr_std = snr_std+((l-total_SNR/self.size)**2)
                snr_std = torch.sqrt(snr_std/(self.size-1))

                tsnr_std = 0
                for l in TSNR_list:
                    tsnr_std = tsnr_std+((l-total_tSNR/self.size)**2)
                tsnr_std = torch.sqrt(tsnr_std/(self.size-1))
                print("Average MSE loss is ", float(total_loss / self.size), "Average SNR is ", float(total_SNR / self.size),
                      "Avearge TSNR is: ", float(total_tSNR/self.size), "Average execution time is ", float(total_time / self.size))
                print("the standard deviation of MSE loss is", float(mse_std))
                print("the standard deviation of SNR is", float(snr_std))
                print("the standard deviation of TSNR is", float(tsnr_std))
                if Disp_param == False:
                    return total_loss / self.size
                if Disp_param == True:
                    return lambdas_cvx, lambdas_ncvx, gammas

    def plot_signals(self, path_signal, path_model):
        name = os.path.split(path_signal)[1]
        name = name.replace('.npy', '')
        x_true = torch.unsqueeze(torch.tensor(
            np.load(path_signal, allow_pickle=True)), 0)
        x_degraded = torch.unsqueeze(torch.tensor(np.load(path_signal.replace(
            'Groundtruth', 'Degraded').replace('_Gr_', '_De_'), allow_pickle=True)), 0)
        x_pred = self.test(path_model=path_model,
                           path_signal=path_signal, save_estimate=False)
        x = np.linspace(0, 200, 2000)
        y = np.linspace(0, 200, 2049)
        plt.figure(figsize=(6, 4.5), dpi=200)
        plt.plot(x, x_true.squeeze().cpu().numpy(), color='black',
                 linestyle='solid', linewidth=1, markersize=1)
        plt.xticks([0, 50, 100, 150, 200], fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([-3, 105])
        plt.xlim([0, 200])
        plt.savefig(self.path_test+'Groundtruth-plot/'+name)
        plt.show()
        plt.close()

        plt.figure(figsize=(6, 4.5), dpi=200)
        name = name.replace('Gr', 'De')
        plt.plot(y, x_degraded.squeeze().cpu().numpy(),
                 color='black',  linestyle='solid', linewidth=1)

        plt.xticks([0, 50, 100, 150, 200], fontsize=20)

        plt.yticks(fontsize=20)
        plt.xlim([0, 200])
        plt.ylim([-80, 80])
#         plt.ylim([-1,103])
        plt.savefig(self.path_test+'Degraded-plot/'+name)
        plt.show()
        plt.close()
        plt.figure(figsize=(6, 4.5), dpi=200)
        name = name.replace('De', 'Es')

        plt.plot(x, x_pred.squeeze().cpu().numpy(),
                 color='black', linestyle='solid', linewidth=1)

        plt.xlim([0, 200])
        plt.xticks([0, 50, 100, 150, 200], fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([-3, 105])

        plt.savefig(os.path.split(path_model)[0]+"/Reconstructed/"+name)
        plt.show()
        plt.close()

    def plot_lambda(self, path_model):
        N = 10  # overparametrization parameter
        if self.mode == "learning_lambda_MM":
            checkpoint = torch.load(path_model)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            list_lambda_k_cvx = []
            list_lambda_k_ncvx = []
            list_gamma_k = []
            if self.architecture_lambda == "lambda_Arch1" or 'lambda_Arch1_cvx' or 'lambda_Arch1_ncvx':
                i = 0
                for name, param in self.model.state_dict().items():

                    if name == 'Layers.'+str(i)+'.Iter.architecture.lamda_cvx':
                        list_lambda_k_cvx.append(FF.relu(param).cpu().numpy())

                    if name == 'Layers.'+str(i)+'.Iter.architecture.lamda_ncvx':
                        list_lambda_k_ncvx.append(FF.relu(param).cpu().numpy())

                    if name == 'Layers.'+str(i)+'.Iter.architecture.gamma':
                        list_gamma_k.append(FF.relu(param).cpu().numpy())
                        i = i+1
            if self.architecture_lambda == "lambda_Arch1_overparam" or 'lambda_Arch1_cvx_overparam' or 'lambda_Arch1_ncvx_overparam':
                lamda_cvx = 1
                lamda_ncvx = 1
                gamma = 1
                i = 0
                j = 1
                k = 1
                l = 1
                for name, param in self.model.state_dict().items():
                    if name == 'Layers.'+str(i)+'.Iter.architecture.lamda_cvx_'+str(j):
                        lamda_cvx = lamda_cvx*param
                        j = j+1
                        if j == N:
                            list_lambda_k_cvx.append(
                                FF.relu(lamda_cvx).cpu().numpy())
                            j = 1
                            lamda_cvx = 1

                    if name == 'Layers.'+str(i)+'.Iter.architecture.lamda_ncvx_'+str(k):
                        lamda_ncvx = lamda_ncvx*param
                        k = k+1
                        if k == N:
                            list_lambda_k_ncvx.append(
                                FF.relu(lamda_ncvx).cpu().numpy())
                            k = 1
                            lamda_ncvx = 1
                    if name == 'Layers.'+str(i)+'.Iter.architecture.gamma_'+str(l):
                        gamma = gamma*param
                        l = l+1
                        if l == N:
                            list_gamma_k.append(FF.relu(gamma).cpu().numpy())
                            l = 1
                            gamma = 1
                            i = i+1

            if self.architecture_lambda == "lambda_Arch2" or 'lambda_Arch2_cvx' or 'lambda_Arch2_ncvx' or "lamda_Arch2_overparam" or "lamda_Arch2_cvx_overparam" or "lamda_Arch2_ncvx_overparam":

                list_lambda_k_cvx, list_lambda_k_ncvx, list_gamma_k = self.test(
                    path_set=self.path_test, path_model=path_model, need_names="no", save_estimate=False, Disp_param=True)

            plt.figure(figsize=(6.7, 4), dpi=200)
            plt.plot(list_lambda_k_cvx, color='red', marker='o',
                     linestyle='solid', linewidth=2, markersize=5)
            plt.ylabel(r'$\lambda_{1,k}$', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(linestyle='-', linewidth=0.5)
            plt.show()
            plt.savefig(os.path.join(os.path.split(path_model)[0], str(os.path.split(
                os.path.split(path_model)[0])[1]), "_lambda_k_cvx.png"), bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(6.7, 4), dpi=200)

            plt.plot(list_lambda_k_ncvx, color='red', marker='o',
                     linestyle='solid', linewidth=2, markersize=5)
            plt.ylabel(r'$\lambda_{2,k}$', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(linestyle='-', linewidth=0.5)
            plt.show()
            plt.savefig(os.path.join(os.path.split(path_model)[0],  str(os.path.split(os.path.split(
                path_model)[0])[1]), "_lambda_k_ncvx.png"), bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(6.7, 4), dpi=200)

            plt.plot(list_gamma_k, color='red', marker='o',
                     linestyle='solid', linewidth=2, markersize=5)
            plt.ylabel(r'$\gamma_{k}$', fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid(linestyle='-', linewidth=0.5)
            plt.show()
            plt.savefig(os.path.join(os.path.split(path_model)[0], str(os.path.split(
                os.path.split(path_model)[0])[1]), "_gamma_k.png"), bbox_inches="tight")
            plt.close()
