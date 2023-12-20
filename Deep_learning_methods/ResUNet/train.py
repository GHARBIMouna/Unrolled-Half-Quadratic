import os
import random
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.data import DataLoader

import model, utilities
from modules import *
from Metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-train_batch', '--train_batch_size', default=50, type=int)
parser.add_argument('-val_batch', '--val_batch_size', default=50, type=int)
parser.add_argument('--Dataset', type=str,
                        help='Dataset number')

parser.add_argument('--network', default='ResUNet', type=str,
                    help='network')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='optimizer')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate (default: 5e-4)', dest='lr')
parser.add_argument('--base-lr', '--base-learning-rate', default=5e-6, type=float,
                    help='base learning rate (default: 5e-6)')
parser.add_argument('--scheduler', default='constant-lr', type=str,
                    help='scheduler')

parser.add_argument('--batch_norm', action='store_true',
                    help='apply batch norm')
parser.add_argument('--spectrum-len', default=2000, type=int,
                    help='spectrum length (default: 2000)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--GPU', default=0, type=int,
                    help='GPU id to use.')






    

    

    
def main_worker( args,device, train_loader, val_loader,test_loader):
    # ----------------------------------------------------------------------------------------
    # Create model(s) and send to device(s)
    # ----------------------------------------------------------------------------------------
    net = model.ResUNet(3, args.batch_norm).float()
    # net.cuda()
    torch.set_default_dtype(torch.float64 )

    # ----------------------------------------------------------------------------------------
    # Define criterion(s), optimizer(s), and scheduler(s)
    # ----------------------------------------------------------------------------------------

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = args.lr,weight_decay=0.01)

    if args.scheduler == "decay-lr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    elif args.scheduler == "multiplicative-lr":
        lmbda = lambda epoch: 0.985
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    elif args.scheduler == "cyclic-lr":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = args.base_lr, max_lr = args.lr, mode = 'triangular2', cycle_momentum = False)
    elif args.scheduler == "one-cycle-lr":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, cycle_momentum = False)
    else: # constant-lr
        scheduler = None

    print('Started Training')
    print('Training Details:')
    print('Network:         {}'.format(args.network))
    print('Epochs:          {}'.format(args.epochs))
    print('Train Batch Size:      {}'.format(args.train_batch_size))
    print('Optimizer:       {}'.format(args.optimizer))
    print('Scheduler:       {}'.format(args.scheduler))
    print('Learning Rate:   {}'.format(args.lr))
    print('Spectrum Length: {}'.format(args.spectrum_len))

    train_loss_epochs=[]
    val_loss_epochs=[]
    for epoch in range(args.epochs):
        train_loss = train(train_loader,device, net, optimizer, scheduler ,criterion, epoch, args)
        val_loss = validate(val_loader,device, net, criterion, args)
        train_loss_epochs.append(train_loss)
        val_loss_epochs.append(val_loss)
        if args.scheduler == "decay-lr" or args.scheduler == "multiplicative-lr":
            scheduler.step()
    print('Finished Training')
    plt.figure()
    plt.yscale('log')
    plt.plot(train_loss_epochs)
    plt.plot(val_loss_epochs)
    plt.legend(["Training","Validation"])
    plt.savefig(os.path.join('Deep_learning_methods', 'ResUNet', 'ResUNet.png'))
    plt.close()
    #test
    loss,snr,tsnr,snr_std,tsnr_std=evaluate(test_loader,device,net,criterion, args)
    print("test loss:",loss)
    print("test SNR:",snr)
    print("test TSNR:",tsnr)
    print("test SNR STD",snr_std)
    print("test TSNR STD",tsnr_std)

def train(dataloader,device, net, optimizer, scheduler, criterion, epoch, args):
    
    batch_time = utilities.AverageMeter('Time', ':6.3f')
    losses = utilities.AverageMeter('Loss', ':.4e')
    progress = utilities.ProgressMeter(len(dataloader), [batch_time, losses], prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (target,inputs) in enumerate(dataloader):
        inputs = inputs.float()
        inputs=np.expand_dims(inputs, axis=1)
        inputs = torch.from_numpy(inputs).to(device)
        target = target.float()
        target=np.expand_dims(target, axis=1)
        target = torch.from_numpy(target).to(device)
        output = net(inputs)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if args.scheduler == "cyclic-lr" or args.scheduler == "one-cycle-lr":
            scheduler.step()   
        losses.update(loss.item(), inputs.size(0)) 

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 400 == 0:
            progress.display(i)
    return losses.avg


def validate(dataloader,device, net, criterion, args):
    batch_time = utilities.AverageMeter('Time', ':6.3f')
    losses = utilities.AverageMeter('Loss', ':.4e')
    progress = utilities.ProgressMeter(len(dataloader), [batch_time, losses], prefix='Validation: ')

    with torch.no_grad():
        end = time.time()
        for  i, (target, inputs) in enumerate(dataloader):
            inputs = inputs.float()
            inputs=np.expand_dims(inputs, axis=1)
            inputs = torch.from_numpy(inputs).to(device)
            target = target.float()
            target=np.expand_dims(target, axis=1)
            target = torch.from_numpy(target).to(device)

            output = net(inputs)

            loss_MSE = criterion(output, target)
            losses.update(loss_MSE.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 400 == 0:
                progress.display(i)

    return losses.avg




def evaluate(dataloader,device, net,criterion, args):
    losses = utilities.AverageMeter('Loss', ':.4e')
    net.eval()
    

    with torch.no_grad():
        snr=0
        tsnr=0
        avg_time=0
        SNR_list=[]
        TSNR_list=[]
        for _, (_,target,inputs) in enumerate(dataloader):
            inputs = inputs.float()
            inputs=np.expand_dims(inputs, axis=1)
            inputs = torch.from_numpy(inputs).to(device)
            target = target.float()
            target=np.expand_dims(target, axis=1)
            target = torch.from_numpy(target).to(device)


            start_time=time.time()
            output = net(inputs)
            end_time=time.time()
            
            loss = criterion(output, target)
            
            snr+=SNR(target,output)
            SNR_list.append(SNR(target,output))
            tsnr+=TSNR(target,output)
            TSNR_list.append(TSNR(target,output))
            losses.update(loss.item(), inputs.size(0))
            avg_time+=end_time-start_time
        snr /= len(dataloader)
        tsnr /= len(dataloader)
        avg_time/=len(dataloader)
        snr_std=0
        for l in SNR_list:
            snr_std=snr_std+((l-snr)**2)
        snr_std=torch.sqrt(snr_std/(len(SNR_list)-1))
                
        tsnr_std=0
        for l in TSNR_list:
            tsnr_std=tsnr_std+((l-tsnr)**2)
        tsnr_std=torch.sqrt(tsnr_std/(len(TSNR_list)-1))
        

        print("Neural Network MSE: {}".format(losses.avg))
        print("Exec avg time", avg_time)
    return losses.avg,snr,tsnr,snr_std,tsnr_std


def CreateLoader(need_names,train_batch_size,val_batch_size,path=None,path_set=None):
        if path_set is not None:
            without_extra = os.path.normpath(path_set)
            last_part = os.path.basename(without_extra)
            if last_part == "training" :
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
            path_train=os.path.join(path, "training")
            path_validation=os.path.join(path, "validation")
            train_data = MyDataset(path_train, need_names)
            train_loader = DataLoader(train_data, train_batch_size, shuffle=True)
            val_data = MyDataset(path_validation, need_names)
            val_loader = DataLoader(val_data, val_batch_size, shuffle=True)
            return train_loader, val_loader



        
        

args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

# torch.cuda.set_device(args.GPU)
torch.set_default_dtype(torch.float64 )
# device = torch.device("cuda")
device = torch.device('cpu')

path= os.path.join('Datasets', args.Dataset)
train_loader,val_loader=CreateLoader(need_names="no",train_batch_size=args.train_batch_size,
                                         val_batch_size=args.val_batch_size,path=path)
         
#test
path_test=os.path.join(path, "test")
test_loader=CreateLoader(need_names="yes",train_batch_size=args.train_batch_size,
                             val_batch_size=args.val_batch_size,path_set=path_test)


main_worker(args,device,train_loader,val_loader,test_loader)
