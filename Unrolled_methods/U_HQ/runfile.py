import os
import argparse
import sys
import torch
import torch.nn as nn  
from network import *


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default="learning_lambda_MM",help="This indicates the mode of training to be deep equilibrium or not")
parser.add_argument('--function', type=str,default="train", help="Function to run: train, test,plot_lambda,plot_signals,plot_learnt_parameters")
parser.add_argument('--number_layers', type=int,default=8)
parser.add_argument('--penalization_cvx', default="fair", type=str,help="Convex penalization")
parser.add_argument('--penalization_ncvx',default="cauchy", type=str,help="Non-convex penalization")
parser.add_argument('--Dataset', type=str,required=True, help="Dataset name to be given")
parser.add_argument('--delta_cvx', type=str, default=0.1, help="smoothing parameter of convex penalization")
parser.add_argument('--delta_ncvx', type=str,default=1,help="smoothing parameter of non-convex penalization")
parser.add_argument('--architecture_lambda', type=str, required=True, help="Name of architecture learning regularization and stepsize")
parser.add_argument('--epochs', type=int, default=150, help="Maximal number of epochs")
parser.add_argument('--lr', type=float, default=10e-2, help="learning rate")
parser.add_argument('--train_batch_size', type=int, default=5, help="training batch size")
parser.add_argument('--val_batch_size', type=int, default=5, help="validation batch size")
parser.add_argument('--test_batch_size', type=int, default=1, help="test batch size")


args = parser.parse_args()

if args.mode not in  ["learning_lambda_MM","Deep_equilibrium"]:
    print("Give valid mode")
    sys.exit(1)
if args.function not  in ['train', 'test','plot_lambda','plot_signals','plot_learnt_parameters']:
    print('Give valid function to execute')
    sys.exit(1)
if args.penalization_cvx not  in ['fair','green',"Tikhonov"]:
    print('Give valid Convex penalization!')
    sys.exit(1)

if args.penalization_ncvx not  in ['green','GMc','welsh','cauchy']:
    print('Give valid Non-Convex penalization!')
    sys.exit(1)



Path=os.path.join(".\Datasets", args.Dataset)
   







V2=torch.eye(2000).type(torch.FloatTensor)# L=Id 
# V2=torch.eye(2000).type(torch.cuda.FloatTensor)# L=Id 
intial_x0="Null_initialization" #Initial x_0 input of the architecture
Initializations=[intial_x0,args.number_layers,V2,args.delta_cvx,args.delta_ncvx]
train_conditions=[args.epochs,args.train_batch_size,args.val_batch_size,args.architecture_lambda]
Folders=[os.path.join(Path,'training'),os.path.join(Path,'validation'),os.path.join(Path,'test'),Path]
test_conditions=[args.test_batch_size]



Network = U_HQ_class(Initializations, train_conditions, test_conditions, args.penalization_cvx,args.penalization_ncvx, Folders, args.mode)


if args.function=='train':
    Network.train(lr=args.lr,number_training="1")
    

if args.function=='test':
    ##In order to test, please give the path of a saved model, e.g:
    path_model='./Datasets/Dataset1/Trained_model1/epoch100'
    Network.test(path_set=os.path.join(Path,'test'),path_model=path_model,need_names="yes")

if args.function=='plot_learnt_parameters':
    ##In order to plot learnt parameters, please give the path of a saved model, e.g:
    path_model='./Datasets/Dataset5/Trained_Model_learning_lambda_MM_1/epoch1'
    Network.plot_lambda(path_model=path_model)

if args.function=='plot_signals':
    ##In order to plot signals, please give the path of a saved model and the path of a specific Groundtruth signal e.g:
    path_signal='./Datasets/Dataset1/training/Groundtruth/x_Gr_te_10.npy'
    path_model='./Datasets/Dataset1/Trained_model1/epoch100'
    Network.plot_signals(path_signal=path_signal,path_model=path_model)
