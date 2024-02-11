import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PD_module import MyDataset
from PD_model import PD_model
from PD_utils import SNR, TSNR
import torch.nn.functional as FF
Loss_fun = nn.MSELoss()


class U_PD(nn.Module):
    def __init__(self, Initialization, train_conditions, paths):
        super(U_PD, self).__init__()
        self.number_layers, self.initial_x0 = Initialization
        self.number_epochs, self.lr, self.train_batch_size, self.val_batch_size, self.test_batch_size, self.dataset = train_conditions
        self.path_train, self.path_validation, self.path_test, self.path_save_model = paths
        self.model = PD_model(self.number_layers).cuda()
        self.dtype = torch.cuda.DoubleTensor

    def CreateLoader(self, need_names):
        train_data = MyDataset(self.path_train, self.initial_x0, need_names)
        self.train_loader = DataLoader(
            train_data, batch_size=self.train_batch_size, shuffle=True)
        val_data = MyDataset(self.path_validation, self.initial_x0, need_names)
        self.val_loader = DataLoader(
            val_data, batch_size=self.val_batch_size, shuffle=True)
        test_data = MyDataset(self.path_test, self.initial_x0, need_names)
        self.test_loader = DataLoader(
            test_data, batch_size=self.test_batch_size, shuffle=True)

    def train(self, number_try, need_names, path_model=None):
        epoc = 0
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if path_model is not None:
            checkpoint = torch.load(path_model)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            epoc = checkpoint['epoch'] + 1
        else:
            if not os.path.exists(os.path.join(self.path_save_model, 'Trained_Model_PD'+'_'+str(number_try))):
                os.makedirs(os.path.join(self.path_save_model,
                            'Trained_Model_PD'+'_'+str(number_try)))
            file_object = open(os.path.join(
                self.path_save_model, 'Trained_Model_PD'+'_'+str(number_try)) + "/readme.txt", "a")
            file_object.writelines(["Dataset:" + str(self.dataset) + '\n',
                                    "Optimizer: " + str(optimizer) + '\n',
                                    "learning_rate:" + str(self.lr) + '\n',
                                    "Number layers: " +
                                    str(self.number_layers) + '\n',
                                    "batch_train_size: " +
                                    str(self.train_batch_size) + '\n',
                                    "batch_val_size: " +
                                    str(self.val_batch_size) + '\n',
                                    ])
            file_object.close()

        self.CreateLoader(need_names=need_names)

        loss_epochs = []
        val_loss_epochs = []

        for epoch in range(self.number_epochs):
            print("Epoch", epoch)
            if epoch+epoc == 0:
                self.model.eval()
            if epoch+epoc > 0:
                self.model.train()
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
                    x_degraded = Variable((x_degraded).type(
                        self.dtype), requires_grad=False)
                    H = Variable((H).type(self.dtype), requires_grad=False)
                    x0 = Variable(torch.zeros((self.train_batch_size, x_true.size()[1])).type(
                        self.dtype), requires_grad=False)
                    x0_d = Variable(torch.zeros((self.train_batch_size, x_degraded.size()[1])).type(
                        self.dtype), requires_grad=False)

                if epoch+epoc == 0:
                    p0 = x0
                    p0_old = x0
                    d0 = x0_d
                    d0_old = x0_d
                    y = x_degraded
                    x_pred, x_pred1 = self.model(H, p0, p0_old, d0, d0_old, y)
                    loss = Loss_fun(x_pred, x_true).detach()
                    snr = SNR(x_true, x_pred).detach()
                    running_loss += loss.item()
                    total_SNR += snr
                if epoch+epoc > 0:
                    optimizer.zero_grad()
                    p0 = x0
                    d0 = x0_d
                    d0_old = x0_d
                    y = x_degraded
                    x_pred, x_pred1 = self.model(H, p0, p0_old, d0, d0_old, y)
                    snr = SNR(x_true, x_pred).detach()
                    loss = Loss_fun(x_pred, x_true)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    total_SNR += snr
                    torch.autograd.set_detect_anomaly(True)
            loss_epochs.append(running_loss / (len(self.train_loader)))
            print("Train AVG MSE for epoch", epoch + epoc, "is", float(running_loss /
                  (len(self.train_loader))), "And Train AVG SNR is", float(total_SNR/(len(self.train_loader))))

            # saving model here
            torch.save({
                'epoch': epoch + epoc,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss, },
                os.path.join(self.path_save_model, 'Trained_Model_PD'+'_'+str(number_try)) + '/epoch' + str(epoch + epoc))

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
                        H = Variable((H).type(self.dtype), requires_grad=False)
                        x0 = Variable(torch.zeros((self.val_batch_size, x_true.size()[1])).type(
                            self.dtype), requires_grad=False)
                        x0_d = Variable(torch.zeros((self.val_batch_size, x_degraded.size()[1])).type(
                            self.dtype), requires_grad=False)

                    p0 = x0
                    d0 = x0_d
                    d0_old = x0_d
                    y = x_degraded
                    x_pred, x_pred1 = self.model(H, p0, p0_old, d0, d0_old, y)
                    loss_val = Loss_fun(x_pred, x_true)
                    snr_val = SNR(x_true, x_pred)
                    loss_current_val += torch.Tensor.item(loss_val)
                    total_SNR_val += snr_val
                val_loss_epochs.append(
                    loss_current_val / (len(self.val_loader)))
                print("Val AVG MSE for epoch", epoch + epoc, "is", float(loss_current_val / (len(
                    self.val_loader))), " And Val Avg SNR is:", float(total_SNR_val/(len(self.val_loader))))
                # Saving training evolution in readme file
                file_object = open(
                    os.path.join(self.path_save_model, 'Trained_Model_PD' +
                                 '_' + str(number_try)) + "/readme.txt",
                    "a")
                file_object.writelines(["Train loss for epoch " + str(epoch + epoc) + "is: " + str(running_loss / (len(self.train_loader)))+'\n', "Train SNR for epoch " + str(epoch + epoc) + "is: " + str(total_SNR / (len(self.train_loader)))+'\n',
                                        "Val loss for epoch" + str(epoch + epoc) + "is: " + str(loss_current_val / (len(self.val_loader))) + '\n', "Val SNR for epoch" + str(
                    epoch + epoc) + "is: " + str(total_SNR_val / (len(self.val_loader))) + '\n'
                ])
                file_object.close()
                plt.plot(loss_epochs, color='black',
                         linestyle='dashed', linewidth=1)
                plt.plot(val_loss_epochs, color='blue',
                         linestyle='dashed', linewidth=1)
                axes = plt.gca()
                axes.set_xlabel('Epochs')
                axes.set_ylabel('Average MSE loss')
                plt.legend(["Training loss", "Validation loss"])
                plt.savefig(os.path.join(self.path_save_model, 'Trained_Model_PD' + '_' + str(
                    number_try))+'/Trained_Model_PD' + str(number_try)+"Training_curves.png")
        return

    def test(self, path_set=None, path_model=None, need_names="no", path_signal=None, save_estimate=False):
        checkpoint = torch.load(path_model, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        if path_signal is not None:

            name = os.path.split(path_signal)[1]
            x_true = torch.unsqueeze(torch.tensor(
                np.load(path_signal, allow_pickle=True)), 0)
            x_degraded = torch.unsqueeze(torch.tensor(np.load(path_signal.replace(
                'Groundtruth', 'Degraded').replace('_Gr_', '_De_'), allow_pickle=True)), 0)
            H = torch.unsqueeze(torch.tensor(np.load(path_signal.replace(
                'Groundtruth', 'H').replace('x_Gr', 'H'), allow_pickle=True)), 0)

            x_true = Variable(x_true.type(self.dtype), requires_grad=False)
            x_degraded = Variable((x_degraded).type(
                self.dtype), requires_grad=False)
            H = Variable((H).type(self.dtype), requires_grad=False)
            if self.initial_x0 == "Null_initialization":
                x_0 = Variable(torch.zeros((1, x_true.size()[1])).type(
                    self.dtype), requires_grad=False)
                x0_d = Variable(torch.zeros((1, x_degraded.size()[1])).type(
                    self.dtype), requires_grad=False)

            p0 = x_0
            p0_old = x_0
            d0 = x0_d
            d0_old = x0_d
            y = x_degraded
            x_pred, x_pred1 = self.model(H, p0, p0_old, d0, d0_old, y)
            loss = Loss_fun(x_pred, x_true).detach()

            snr = SNR(x_true, x_pred)
            print("loss is:", loss, "and SNR is", snr)
            return x_pred.detach()

        else:

            self.CreateLoader(need_names)
            total_loss = 0
            total_time = 0
            total_SNR = 0
            total_tSNR = 0

            MSE_list = []
            SNR_list = []
            TSNR_list = []

            for minibatch in self.test_loader:
                if self.initial_x0 == "Null_initialization":
                    if need_names == "yes":
                        name, H, x_true, x_degraded = minibatch
                        name = name[0]
                    if need_names == "no":

                        [H, x_true, x_degraded] = minibatch
                    x_true = Variable(x_true.type(
                        self.dtype), requires_grad=False)
                    H = Variable(H.type(self.dtype), requires_grad=False)
                    x_degraded = Variable((x_degraded).type(
                        self.dtype), requires_grad=False)
                    x0_d = Variable(torch.zeros((self.test_batch_size, x_degraded.size()[1])).type(
                        self.dtype), requires_grad=False)
                    x_0 = Variable(torch.zeros((self.test_batch_size, x_true.size()[1])).type(
                        self.dtype), requires_grad=False)

                t0 = time.time()
                p0 = x_0
                p0_old = x_0
                d0 = x0_d
                d0_old = x0_d
                y = x_degraded
                x_pred, x_pred1 = self.model(H, p0, p0_old, d0, d0_old, y)
                t1 = time.time()
                loss = (Loss_fun(x_true, x_pred).detach())
                snr = SNR(x_true, x_pred).detach()
                tsnr = TSNR(x_true, x_pred).detach()
                MSE_list.append(loss)
                SNR_list.append(snr)
                TSNR_list.append(tsnr)

                total_loss += loss
                total_time += t1-t0
                total_SNR += snr
                total_tSNR += tsnr

            # compute metrics STD
            mse_std = 0
            for l in MSE_list:
                mse_std = mse_std+((l-total_loss/len(self.test_loader))**2)
            mse_std = torch.sqrt(mse_std/(len(self.test_loader)-1))

            snr_std = 0
            for l in SNR_list:
                snr_std = snr_std+((l-total_SNR/len(self.test_loader))**2)
            snr_std = torch.sqrt(snr_std/(len(self.test_loader)-1))

            tsnr_std = 0
            for l in TSNR_list:
                tsnr_std = tsnr_std+((l-total_tSNR/len(self.test_loader))**2)
            tsnr_std = torch.sqrt(tsnr_std/(len(self.test_loader)-1))

            print("Average Test MSE loss is ", float(total_loss / len(self.test_loader)), "Average Test SNR is ", float(total_SNR / len(self.test_loader)),
                  "Avearge Test TSNR is: ", float(total_tSNR/len(self.test_loader)), "Average execution time is ", total_time / len(self.test_loader))
            print("the standard deviation of MSE loss is", float(mse_std))
            print("the standard deviation of SNR is", float(snr_std))
            print("the standard deviation of TSNR is", float(tsnr_std))
            return

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

        plt.figure()
        plt.plot(x, x_true.squeeze().cpu().numpy(), color='black',
                 linestyle='solid', linewidth=0.5, markersize=1)
        plt.xticks([0, 50, 100, 150, 200], fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([-3, 105])
        plt.xlim([0, 200])
        plt.savefig(self.path_save_model+'Groundtruth-plot/'+name)
        plt.show()
        plt.close()

        plt.figure()
        name = name.replace('Gr', 'De')
        plt.plot(y, x_degraded.squeeze().cpu().numpy(),
                 color='black',  linestyle='solid', linewidth=0.5)
        plt.xticks([0, 50, 100, 150, 200], fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim([0, 200])
        plt.ylim([-80, 80])
        plt.savefig(self.path_save_model+'Degraded-plot/'+name)
        plt.show()
        plt.close()

        plt.figure()
        name = name.replace('De', 'Es')
        plt.plot(y, x_pred.squeeze().cpu().numpy(),
                 color='black', linestyle='solid', linewidth=0.5)
        plt.xlim([0, 200])
        plt.xticks([0, 50, 100, 150, 200], fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim([-3, 105])
        plt.savefig(self.path_save_model+"/Reconstructed/"+name)
        plt.show()
        plt.close()

    def plot_stepsizes(self, path_model):
        checkpoint = torch.load(path_model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        i = 0
        list_sigma = []
        list_tau = []
        list_rho = []
        for name, param in self.model.state_dict().items():
            if name == 'Layers.'+str(i)+'.architecture.tau':
                list_tau.append(FF.relu(param).cpu().numpy())
            if name == 'Layers.'+str(i)+'.architecture.sigma':
                list_sigma.append(FF.relu(param).cpu().numpy())
            if name == 'Layers.'+str(i)+'.architecture.rho':
                list_rho.append(FF.relu(param).cpu().numpy())
                i = i+1

        plt.figure()
        plt.plot(list_sigma, color='red', marker='o',
                 markersize=2, linewidth=0.5, linestyle="--")
        plt.ylabel(r'$\gamma_k$', fontsize=18)
        plt.xlabel(r'layer $k$', fontsize=18)
        plt.grid(linestyle='-', linewidth=0.5)
        plt.show()
        plt.savefig(os.path.split(path_model)[
                    0]+"/" + str(os.path.split(os.path.split(path_model)[0])[1])+"learnt_sigma_k.png")
        plt.close()

        plt.figure()
        plt.plot(list_tau, color='red', marker='o',
                 markersize=2, linewidth=0.5, linestyle="--")
        plt.ylabel(r'$\tau_k$', fontsize=18)
        plt.xlabel(r'layer $k$', fontsize=18)
        plt.grid(linestyle='-', linewidth=0.5)
        plt.show()
        plt.savefig(os.path.split(path_model)[
                    0]+"/" + str(os.path.split(os.path.split(path_model)[0])[1])+"learnt_tau_k.png")
        plt.close()

        plt.figure()
        plt.plot(list_tau, color='red', marker='o',
                 markersize=2, linewidth=0.5, linestyle="--")
        plt.plot([0.1]*self.number_layers, color='black', marker='o',
                 markersize=2, linewidth=0.5, linestyle="--")
        plt.ylabel(r'$\rho_k$', fontsize=18)
        plt.xlabel(r'layer $k$', fontsize=18)
        plt.grid(linestyle='-', linewidth=0.5)
        plt.show()
        plt.savefig(os.path.split(path_model)[
                    0]+"/" + str(os.path.split(os.path.split(path_model)[0])[1])+"learnt_rho_k.png")
        plt.close()
