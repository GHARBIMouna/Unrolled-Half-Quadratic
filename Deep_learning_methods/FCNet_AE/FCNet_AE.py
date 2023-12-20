import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import os
import time
from matplotlib import pyplot as plt
from modules import *
from Metrics import *


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(2049, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 2000)
        self.fc4 = nn.Linear(2000, 2000)

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        return x


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(2049, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2000),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def CreateLoader(need_names, train_batch_size, val_batch_size, path=None, path_set=None):
    if path_set is not None:
        without_extra = os.path.normpath(path_set)
        last_part = os.path.basename(without_extra)
        if last_part == "training":
            train_data = MyDataset(path_set, need_names)
            loader = DataLoader(train_data, batch_size=1, shuffle=False)
        if last_part == "validation":
            val_data = MyDataset(path_set, need_names)
            loader = DataLoader(val_data, batch_size=1, shuffle=False)
        if last_part == "test":
            test_data = MyDataset(path_set, need_names)
            loader = DataLoader(test_data, batch_size=1, shuffle=False)
        return loader

    else:

      # For training purposes
        path_train = os.path.join(path, "training")
        path_validation = os.path.join(path, "validation")
        train_data = MyDataset(path_train, need_names)
        train_loader = DataLoader(train_data, train_batch_size, shuffle=True)
        val_data = MyDataset(path_validation, need_names)
        val_loader = DataLoader(val_data, val_batch_size, shuffle=True)
        return train_loader, val_loader


def train(args, device, model, train_loader, optimizer, epoch):
    train_loss = 0
    model.train()

    for batch_idx, (target, data) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        train_loss += nn.MSELoss()(output, target).item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    return train_loss


def val(device, model, val_loader):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for target, data in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += nn.MSELoss()(output, target).item()

    val_loss /= len(val_loader)
    print('\nTest set: Average loss: {:.4f}  \n'.format(val_loss))
    return val_loss


def test(device, model, test_loader):
    model.eval()
    loss = 0
    snr = 0
    tsnr = 0
    avg_time = 0
    SNR_list = []
    TSNR_list = []

    with torch.no_grad():
        for name, target, data in test_loader:
            data, target = data.to(device), target.to(device)
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            loss += nn.MSELoss()(output, target).item()
            snr += SNR(target, output)
            SNR_list.append(SNR(target, output))
            tsnr += TSNR(target, output)
            TSNR_list.append(TSNR(target, output))
            avg_time += end_time-start_time

    loss /= len(test_loader)
    snr /= len(test_loader)
    tsnr /= len(test_loader)
    avg_time /= len(test_loader)

    snr_std = 0
    for l in SNR_list:
        snr_std = snr_std+((l-snr)**2)
    snr_std = torch.sqrt(snr_std/(len(SNR_list)-1))

    tsnr_std = 0
    for l in TSNR_list:
        tsnr_std = tsnr_std+((l-tsnr)**2)
    tsnr_std = torch.sqrt(tsnr_std/(len(TSNR_list)-1))

    print("average test inference time is:", avg_time)
    return loss, snr, tsnr, snr_std, tsnr_std


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--Architecture', type=str,
                        help='FCNet or AE')
    parser.add_argument('--train_batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--val_batch_size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 50)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--GPU', type=int, default=0,
                        help='GPU number (default: 0)')
    parser.add_argument('--Dataset', type=str,
                        help='Dataset number')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.GPU)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda")

    path = os.path.join("Datasets", args.Dataset)

    train_loader, val_loader = CreateLoader(
        need_names="no", train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, path=path)

    if args.Architecture == "FCNet":
        model = FCNet().cuda()
    elif args.Architecture == "AE":
        model = AE().cuda()
    else:
        print("Give valid architecture name.")
        sys.exit(1)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    train_loss_epochs = []
    val_loss_epochs = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, device, model, train_loader, optimizer, epoch)
        val_loss = val(device, model, val_loader)
        train_loss_epochs.append(train_loss)
        val_loss_epochs.append(val_loss)
        scheduler.step()

    plt.figure()
    plt.plot(train_loss_epochs)
    plt.plot(val_loss_epochs)
    plt.savefig(os.path.join('Deep_learning_methods', 'FCNet_AE', 'FCNet.png'))
    plt.close()

    # test
    path_test = os.path.join(path, "test")
    test_loader = CreateLoader(need_names="yes", train_batch_size=args.train_batch_size,
                               val_batch_size=args.val_batch_size, path_set=path_test)
    loss, snr, tsnr, snr_std, tsnr_std = test(device, model, test_loader)

    print("test loss:", loss)
    print("test SNR:", snr)
    print("test SNR STD", snr_std)
    print("test TSNR:", tsnr)
    print("test TSNR STD", tsnr_std)


if __name__ == '__main__':
    main()
