import argparse
import os
import sys
import numpy as np
from Utils import *
from Create_p_z import *


parser = argparse.ArgumentParser()
parser.add_argument('--Dataset', type=str, required=True,
                    help="Dataset name to be given")
parser.add_argument('--N_tr', type=int,
                    default=1000, help="training set size")
parser.add_argument('--N_val', type=int,
                    default=200, help="validation set size")
parser.add_argument('--N_test', type=int,
                    default=200, help="test set size")
parser.add_argument('--a', type=float, default=0.2,
                    help="asymmetry coefficient of FS Kernel: 0.2, 0.4, 0.6")
parser.add_argument('--percentage', type=float, default=0.03,
                    help="Percentage of spikes wrt original signal size: 0.015, 0.03, 0.045")
args = parser.parse_args()
parser.add_argument('--peak_inter_dist', type=int, default=5,
                    help="distance between peaks: 1, 3, 5")
parser.add_argument('--sG', type=float, default=0.02,
                    help="additive noise variance: 0.02, 0.04, 0.06")


assert (args.Dataset)
datasetRoot = os.path.join('./Chromatography_Toolbox', args.Dataset)

if os.path.exists(datasetRoot):
    print('ERROR: dataset already exists. Delete to re-create or choose a different path.')
    sys.exit(1)

os.makedirs(datasetRoot)


training_path = os.path.join(datasetRoot, "training")
os.makedirs(training_path)

training_path_Groundtruth = os.path.join(training_path, "Groundtruth")
os.makedirs(training_path_Groundtruth)
training_path_Degraded = os.path.join(training_path, "Degraded")
os.makedirs(training_path_Degraded)

training_path_Infos = os.path.join(training_path, "Infos")
os.makedirs(training_path_Infos)

training_path_Infos_H = os.path.join(training_path_Infos, "H")
os.makedirs(training_path_Infos_H)

training_path_Infos_L = os.path.join(training_path_Infos, "L")
os.makedirs(training_path_Infos_L)

training_path_Infos_Supp_Eff = os.path.join(training_path_Infos, "Supp-Eff")
os.makedirs(training_path_Infos_Supp_Eff)


validation_path = os.path.join(datasetRoot, "validation")
os.makedirs(validation_path)
validation_path_Groundtruth = os.path.join(
    validation_path, "Groundtruth")
os.makedirs(validation_path_Groundtruth)
validation_path_Degraded = os.path.join(validation_path, "Degraded")
os.makedirs(validation_path_Degraded)


validation_path_Infos = os.path.join(validation_path, "Infos")
os.makedirs(validation_path_Infos)

validation_path_Infos_H = os.path.join(validation_path_Infos, "H")
os.makedirs(validation_path_Infos_H)

validation_path_Infos_L = os.path.join(validation_path_Infos, "L")
os.makedirs(validation_path_Infos_L)

validation_path_Infos_Supp_Eff = os.path.join(
    validation_path_Infos, "Supp-Eff")
os.makedirs(validation_path_Infos_Supp_Eff)


test_path = os.path.join(datasetRoot, "test")
os.makedirs(test_path)

test_path_Groundtruth = os.path.join(test_path, "Groundtruth")
os.makedirs(test_path_Groundtruth)
test_path_Degraded = os.path.join(test_path, "Degraded")
os.makedirs(test_path_Degraded)


test_path_Infos = os.path.join(test_path, "Infos")
os.makedirs(test_path_Infos)

test_path_Infos_H = os.path.join(test_path_Infos, "H")
os.makedirs(test_path_Infos_H)

test_path_Infos_L = os.path.join(test_path_Infos, "L")
os.makedirs(test_path_Infos_L)

test_path_Infos_Supp_Eff = os.path.join(test_path_Infos, "Supp-Eff")
os.makedirs(test_path_Infos_Supp_Eff)


##### Parameters of the Fraser Suzuki kernel#####
m = 0
sigma = 0.5
T = np.arange(m-(sigma/args.a)+0.0001, 20, step=0.2)
T_f = Fraser_Suzuki(T, args.a, m, sigma)
################################################
N = 2000  # size of initial signal
n = int(args.percentage*N)  # number of spikes
extrem_dist = len(T)


########## Parameters of blurring Gaussian kernel and noise#########
############# Settings of blurring Gaussain kernel##################
Nh = 10
T_h = np.arange(-Nh, Nh, step=0.2)
m2 = 0
s2 = 1
T_h = Gaussian(T_h, m2, s2)
np.save(os.path.join(datasetRoot, "H"), convmtx(T_h, N))
####################################################################
################## Settings of additive Gaussian noise###############
m1 = 0
s1 = args.sG

#########################################
###### Dataset settings###################
x_temp = []
for j in range(args.N_tr):
    mode = 'training'
    x_true, H_Gr, L_Gr, Supp_eff_mat_Gr = x_creation(
        N, n, extrem_dist, args.peak_inter_dist, T_f)
    x_degraded = y_degradation(x_true, m1, s1, T_h)
    x_temp.append(x_true)
    while j > 0 and np.array_equal(x_true, x_temp[-2]):
        print('Two created signals are equal, deleting and creating new signal!')
        x_true, H_Gr, L_Gr, Supp_eff_mat_Gr = x_creation(
            N, n, extrem_dist, args.peak_inter_dist, T_f)
        x_degraded = y_degradation(x_true, m1, s1, T_h)
        x_temp.pop()
        x_temp.append(x_true)

    np.save(os.path.join(training_path_Groundtruth,
            "x_Gr_"+mode[0:2]+"_"+str(j)), x_true.ravel())
    np.save(os.path.join(training_path_Degraded, "x_De_" +
            mode[0:2]+"_"+str(j)), x_degraded.ravel())
    np.save(os.path.join(training_path_Infos_H,
            "x_In_H_"+mode[0:2]+"_"+str(j)), H_Gr.ravel())
    np.save(os.path.join(training_path_Infos_L,
            "x_In_L_"+mode[0:2]+"_"+str(j)), L_Gr.ravel())
    np.save(os.path.join(training_path_Infos_Supp_Eff,
            "x_In_SE_"+mode[0:2]+"_"+str(j)), Supp_eff_mat_Gr)


for j in range(args.N_val):
    mode = 'validation'
    x_true, H_Gr, L_Gr, Supp_eff_mat_Gr = x_creation(
        N, n, extrem_dist, args.peak_inter_dist, T_f)
    x_degraded = y_degradation(x_true, m1, s1, T_h)
    while np.array_equal(x_true, x_temp[-2]):
        print('Two created signals are equal, deleting and creating new signal!')
        x_true, H_Gr, L_Gr, Supp_eff_mat_Gr = x_creation(
            N, n, extrem_dist, args.peak_inter_dist, T_f)
        x_degraded = y_degradation(x_true, m1, s1, T_h)
        x_temp.pop()
        x_temp.append(x_true)
    np.save(os.path.join(validation_path_Groundtruth,
            "x_Gr_"+mode[0:2]+"_"+str(j)), x_true.ravel())
    np.save(os.path.join(validation_path_Degraded, "x_De_" +
            mode[0:2]+"_"+str(j)), x_degraded.ravel())
    np.save(os.path.join(validation_path_Infos_H,
            "x_In_H_"+mode[0:2]+"_"+str(j)), H_Gr.ravel())
    np.save(os.path.join(validation_path_Infos_L,
            "x_In_L_"+mode[0:2]+"_"+str(j)), L_Gr.ravel())
    np.save(os.path.join(validation_path_Infos_Supp_Eff,
            "x_In_SE_"+mode[0:2]+"_"+str(j)), Supp_eff_mat_Gr)


for j in range(args.N_test):
    mode = 'test'
    x_true, H_Gr, L_Gr, Supp_eff_mat_Gr = x_creation(
        N, n, extrem_dist, args.peak_inter_dist, T_f)
    x_degraded = y_degradation(x_true, m1, s1, T_h)
    x_temp.append(x_true)
    while np.array_equal(x_true, x_temp[-2]):
        print('Two created signals are equal, deleting and creating new signal!')
        x_true, H_Gr, L_Gr, Supp_eff_mat_Gr = x_creation(
            N, n, extrem_dist, args.peak_inter_dist, T_f)
        x_degraded = y_degradation(x_true, m1, s1, T_h)
        x_temp.pop()
        x_temp.append(x_true)
    np.save(os.path.join(test_path_Groundtruth, "x_Gr_" +
            mode[0:2]+"_"+str(j)), x_true.ravel())
    np.save(os.path.join(test_path_Degraded, "x_De_" +
            mode[0:2]+"_"+str(j)), x_degraded.ravel())
    np.save(os.path.join(test_path_Infos_H, "x_In_H_" +
            mode[0:2]+"_"+str(j)), H_Gr.ravel())
    np.save(os.path.join(test_path_Infos_L, "x_In_L_" +
            mode[0:2]+"_"+str(j)), L_Gr.ravel())
    np.save(os.path.join(test_path_Infos_Supp_Eff,
            "x_In_SE_"+mode[0:2]+"_"+str(j)), Supp_eff_mat_Gr)
