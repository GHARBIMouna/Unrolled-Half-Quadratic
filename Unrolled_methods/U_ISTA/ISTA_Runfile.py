import argparse
import os
import sys
from ISTA_Network import *

parser = argparse.ArgumentParser()
parser.add_argument('--function', type=str, default="train",
                    help="Function to run: train, test,plot_signals,plot_stepsizes")
parser.add_argument('--Dataset', type=str, required=True,
                    help="Dataset name to be given")
parser.add_argument('--number_layers', type=int, default=16)
parser.add_argument('--epochs', type=int, default=1500,
                    help="Maximal number of epochs")
parser.add_argument('--lr', type=float, default=10e-3, help="learning rate")
parser.add_argument('--train_batch_size', type=int,
                    default=150, help="training batch size")
parser.add_argument('--val_batch_size', type=int,
                    default=10, help="validation batch size")
parser.add_argument('--test_batch_size', type=int,
                    default=1, help="test batch size")

args = parser.parse_args()

if args.function not in ['train', 'test', 'plot_signals', 'plot_stepsizes']:
    print('Give valid function to execute')
    sys.exit(1)

# train conditions
train_conditions = [args.epochs, args.lr, args.train_batch_size,
                    args.val_batch_size, args.test_batch_size]
Path = os.path.join(".\Datasets", args.Dataset)
paths = [os.path.join(Path, 'training'), os.path.join(
    Path, 'validation'), os.path.join(Path, 'test'), Path]


# Initialization
initial_x0 = "Null_initialization"
Initialization = [args.number_layers, initial_x0]


Net = U_ISTA_class(Initialization, train_conditions, paths)

if args.function == 'train':
    Net.train(number_try="1", need_names="no")

if args.function == 'test':
    # In order to test on the test set and not on one signal, please fill in path_set and give the path of a saved model, e.g:
    path_model = './Datasets/Dataset1/Trained_model1/epoch100'
    Net.test(path_set=os.path.join(Path, 'test'),
             path_model=path_model, need_names="yes")

if args.function == 'plot_signals':
    # In order to plot signals, please give the path of a saved model and the path of a specific Groundtruth signal e.g:
    path_signal = './Datasets/Dataset1/training/Groundtruth/x_Gr_te_10.npy'
    path_model = './Datasets/Dataset1/Trained_model1/epoch100'
    Net.plot_signals(path_signal=path_signal, path_model=path_model)

if args.function == 'plot_stepsizes':
    # In order to plot learnt parameters, please give the path of a saved model, e.g:
    path_model = './Datasets/Dataset1/Trained_Model1/epoch1'
    Net.plot_stepsizes(path_model=path_model)
