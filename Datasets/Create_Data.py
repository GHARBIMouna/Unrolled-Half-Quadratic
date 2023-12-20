import sys
import argparse
import numpy as np
import pandas as pd
import ast
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import os


parser = argparse.ArgumentParser()
parser.add_argument('--Dataset', type=str,
                    help='Give the name of the dataset to create the training, validation and test folders')
parser.add_argument('--Kernel', type=str, default="Ricker",
                    help="Choose the kernel type: Gaussian, Ricker, Variable_Ricker or Variable_Fraser_Suzuki")
parser.add_argument('--Range', type=str, default="Range_1",
                    help="chooses the monoisotopic range of the groundtruth signal, between 0 and 200")
parser.add_argument('--set_size_training', type=int, default=900)
parser.add_argument('--set_size_validation', type=int, default=100)
parser.add_argument('--set_size_test', type=int, default=100)
args = parser.parse_args()


def convmtx(h, n_in):
    """
    Generates a convolution matrix H
    such that the product of H and an i_n element vector
    x is the convolution of x and h.

    Usage: H = convm(h,n_in)
    Given a column vector h of length N, an (N+n_in-1 x n_in)_in convolution matrix is
    generated

    This method has the same functionning as that of np.convolve(x,h)
    """
    N = len(h)
    N1 = N + 2*n_in - 2
    hpad = np.concatenate([np.zeros(n_in-1), h[:], np.zeros(n_in-1)])

    H = np.zeros((len(h)+n_in-1, n_in))
    for i in range(n_in):
        H[:, i] = hpad[n_in-i-1:N1-i]
    return H


def Fraser_Suzuki(t, a, m, sigma):
    """
    Creates a Frazer Suzuki shaped kernel.

    Parameters:
    ----------
    t:     list of input values
    a:     peak tailing/ asymmetry
    m:     mean
    sigma: peak width
    """
    return 1/(sigma * np.sqrt(2 * np.pi)*np.exp((np.log(2)*a**2)/4))*np.exp((-1/(2*a**2)) * np.log(1 + a*(t-m)/sigma)**2)


def Test(molecules, molecules_list):
    """
    Tests for redundancies while creating the Dataset
    """
    if len(molecules_list) == 1:
        return False
    if len(molecules_list) > 1:
        molecules_list_temp = molecules_list[0:-1]

        for element in molecules_list_temp:
            if collections.Counter(element) == collections.Counter(molecules):
                return True
        return False


def create_signal(h, molecule, m1, s1, Range):
    """
    Creates and returns a groundtruth and a degraded signal
    """
    e = 10e-10
    peaks = ast.literal_eval(str(MSdata['peak'][molecule]).replace("'", ""))
    intensities = ast.literal_eval(
        str(MSdata['intensitÈ relative'][molecule]).replace("'", ""))

    if Range == "Range_1":
        n_points = 2000
        Signal0 = np.zeros((n_points))
    for k in range(len(peaks)):
        if Range == "Range_1":
            # to create signals between 0 and 200 that turn between 0 and 2000
            Signal0[int(10*peaks[k])] = intensities[k]/10

    x_true = Signal0
    x_degraded = np.convolve(x_true, h)
    x_degraded_2000 = np.convolve(x_true, h, mode="same")
    p = np.shape(x_degraded)
    # Varaible noise
    s = np.random.uniform(0.0+e, s1)
    s1 = s
    noise = np.random.normal(m1, s1, p)
    x_degraded = x_degraded+noise
    return x_true, x_degraded, x_degraded_2000


assert (args.Dataset)
datasetRoot = os.path.join('./Datasets', args.Dataset)

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
training_path_Degraded2000 = os.path.join(
    training_path, "Degraded2000")
os.makedirs(training_path_Degraded2000)
training_path_H = os.path.join(training_path, "H")
os.makedirs(training_path_H)
validation_path = os.path.join(datasetRoot, "validation")
os.makedirs(validation_path)
validation_path_Groundtruth = os.path.join(
    validation_path, "Groundtruth")
os.makedirs(validation_path_Groundtruth)
validation_path_Degraded = os.path.join(validation_path, "Degraded")
os.makedirs(validation_path_Degraded)
validation_path_Degraded2000 = os.path.join(
    validation_path, "Degraded2000")
os.makedirs(validation_path_Degraded2000)
validation_path_H = os.path.join(validation_path, "H")
os.makedirs(validation_path_H)

test_path = os.path.join(datasetRoot, "test")
os.makedirs(test_path)

test_path_Groundtruth = os.path.join(test_path, "Groundtruth")
os.makedirs(test_path_Groundtruth)
test_path_Degraded = os.path.join(test_path, "Degraded")
os.makedirs(test_path_Degraded)
test_path_Degraded2000 = os.path.join(test_path, "Degraded2000")
os.makedirs(test_path_Degraded2000)
test_path_H = os.path.join(test_path, "H")
os.makedirs(test_path_H, exist_ok=True)


# mean and standard deviation for Gaussian noise additive
m1 = 0
s1 = 0.5

e = 10e-10  # small constant
n_in = 2000  # Size of groundtruth signal

# Creating a database with one chemical element per signal
MSdata = pd.read_csv('./Datasets/MSdata.csv', encoding='mac_roman')


if args.Kernel == "Gaussian":
    # filter size
    Nh = 15
    h = np.linspace(-Nh, Nh)
    # Gaussian kernel
    m2 = 0
    s2 = 1
    h = 1/(s2 * np.sqrt(2 * np.pi)) * np.exp(- (h - m2)**2 / (2 * s2**2))

if args.Kernel == "Ricker":
    # filter size
    Nh = 15
    h = np.linspace(-Nh, Nh)
    # Ricker Kernel
    sigma = 0.5
    h = 2/(np.sqrt(3*sigma)*(np.pi)**0.25) * \
        (1-(h/sigma)**2)*np.exp(-(h**2)/(2*sigma**2))


# Create Dataset with m/z between 0 and 200 (called Range_1)
if args.Range == "Range_1":
    molecules_list = []

    training_counter = 0
    while training_counter < args.set_size_training:
        molecule_number = random.randint(0, len(MSdata)-1)
        molecule = MSdata['Formule chimique'][molecule_number]
        molecules_list.append(molecule)

        while (Test(molecule, molecules_list) == True) or (np.amax(ast.literal_eval(str(MSdata['peak'][molecule_number]).replace("'", ""))) >= 200) or (np.count_nonzero(ast.literal_eval(str(MSdata['peak'][molecule_number]).replace("'", ""))) < 10):
            molecules_list.pop()
            molecule_number = random.randint(0, len(MSdata)-1)
            molecule = MSdata['Formule chimique'][molecule_number]
            molecules_list.append(molecule)

        # Creating variable Kernel
        if args.Kernel == "Variable_Gaussian":
            # filter size
            Nh = 15
            h = np.linspace(-Nh, Nh)
            # Gaussian kernel
            m2 = 0
            s2 = np.random.uniform(0.5+e, 1.5)
            h = 1/(s2 * np.sqrt(2 * np.pi)) * \
                np.exp(- (h - m2)**2 / (2 * s2**2))

        if args.Kernel == "Variable_Ricker":
            # filter size
            Nh = 15
            h = np.linspace(-Nh, Nh)
            # Ricker Kernel
            sigma = np.random.uniform(0.25+e, 1)
            h = 2/(np.sqrt(3*sigma)*(np.pi)**0.25) * \
                (1-(h/sigma)**2)*np.exp(-(h**2)/(2*sigma**2))

        if args.Kernel == "Variable_Fraser_Suzuki":
            m = 0
            a = np.random.uniform(0.2+e, 0.6)
            sigma = np.random.uniform(0.25+e, 1)

            Nh = 15
            h = np.linspace(-Nh, Nh)

            for i in range(len(h)):
                if h[i] < m-(sigma/a):
                    h[i] = 0
                else:
                    pass
            T_f = Fraser_Suzuki(h[h != 0], a, m, sigma)
            T_f = list(h[h == 0])+list(T_f)
            h = T_f

        x_true, x_degraded, x_degraded_2000 = create_signal(
            h, molecule_number, m1, s1, args.Range)
        if np.max(x_true) > 30:
            print("num sig training", training_counter)
            H = convmtx(h, n_in)

            np.save(os.path.join(training_path_H,
                    f'H_tr_{training_counter}'), H)
            np.save(os.path.join(training_path_Groundtruth,
                    f'x_Gr_tr_{training_counter}'), x_true.ravel())
            np.save(os.path.join(training_path_Degraded,
                    f'x_De_tr_{training_counter}'), x_degraded.ravel())
            np.save(os.path.join(training_path_Degraded2000,
                    f'x_De_tr_{training_counter}'), x_degraded_2000.ravel())

            training_counter += 1

    validation_counter = 0
    while validation_counter < args.set_size_validation:
        molecule_number = random.randint(0, len(MSdata)-1)
        molecule = MSdata['Formule chimique'][molecule_number]
        molecules_list.append(molecule)

        while (Test(molecule, molecules_list) == True) or (np.amax(ast.literal_eval(str(MSdata['peak'][molecule_number]).replace("'", ""))) >= 200) or (np.count_nonzero(ast.literal_eval(str(MSdata['peak'][molecule_number]).replace("'", ""))) < 10):
            molecules_list.pop()
            molecule_number = random.randint(0, len(MSdata)-1)
            molecule = MSdata['Formule chimique'][molecule_number]
            molecules_list.append(molecule)

        # Creating variable Kernel
        if args.Kernel == "Variable_Gaussian":
            # filter size
            Nh = 15
            h = np.linspace(-Nh, Nh)
            # Gaussian kernel
            m2 = 0
            s2 = np.random.uniform(0.5+e, 1.5)
            h = 1/(s2 * np.sqrt(2 * np.pi)) * \
                np.exp(- (h - m2)**2 / (2 * s2**2))

        if args.Kernel == "Variable_Ricker":
            # filter size
            Nh = 15
            h = np.linspace(-Nh, Nh)
            # Ricker Kernel
            sigma = np.random.uniform(0.25+e, 1)
            h = 2/(np.sqrt(3*sigma)*(np.pi)**0.25) * \
                (1-(h/sigma)**2)*np.exp(-(h**2)/(2*sigma**2))

        if args.Kernel == "Variable_Fraser_Suzuki":
            m = 0
            a = np.random.uniform(0.2+e, 0.6)
            sigma = np.random.uniform(0.25+e, 1)

            Nh = 15
            h = np.linspace(-Nh, Nh)

            for i in range(len(h)):
                if h[i] < m-(sigma/a):
                    h[i] = 0
                else:
                    pass
            T_f = Fraser_Suzuki(h[h != 0], a, m, sigma)
            T_f = list(h[h == 0])+list(T_f)

            h = T_f

        x_true, x_degraded, x_degraded_2000 = create_signal(
            h, molecule_number, m1, s1, args.Range)
        if np.max(x_true) > 30:
            print("num sig validation", validation_counter)
            H = convmtx(h, n_in)

            np.save(os.path.join(validation_path_H,
                    f'H_va_{validation_counter}'), H)
            np.save(os.path.join(validation_path_Groundtruth,
                    f'x_Gr_va_{validation_counter}'), x_true.ravel())
            np.save(os.path.join(validation_path_Degraded,
                    f'x_De_va_{validation_counter}'), x_degraded.ravel())
            np.save(os.path.join(validation_path_Degraded2000,
                    f'x_De_va_{validation_counter}'), x_degraded_2000.ravel())

            validation_counter += 1

    test_counter = 0
    while test_counter < args.set_size_test:
        molecule_number = random.randint(0, len(MSdata)-1)
        molecule = MSdata['Formule chimique'][molecule_number]
        molecules_list.append(molecule)

        while (Test(molecule, molecules_list) == True) or (np.amax(ast.literal_eval(str(MSdata['peak'][molecule_number]).replace("'", ""))) >= 200) or (np.count_nonzero(ast.literal_eval(str(MSdata['peak'][molecule_number]).replace("'", ""))) < 10):
            molecules_list.pop()
            molecule_number = random.randint(0, len(MSdata)-1)
            molecule = MSdata['Formule chimique'][molecule_number]
            molecules_list.append(molecule)

        # Creating variable Kernel
        if args.Kernel == "Variable_Gaussian":
            # filter size
            Nh = 15
            h = np.linspace(-Nh, Nh)
            # Gaussian kernel
            m2 = 0
            s2 = np.random.uniform(0.5+e, 1.5)
            h = 1/(s2 * np.sqrt(2 * np.pi)) * \
                np.exp(- (h - m2)**2 / (2 * s2**2))

        if args.Kernel == "Variable_Ricker":
            # filter size
            Nh = 15
            h = np.linspace(-Nh, Nh)
            # Ricker Kernel
            sigma = np.random.uniform(0.25+e, 1)
            h = 2/(np.sqrt(3*sigma)*(np.pi)**0.25) * \
                (1-(h/sigma)**2)*np.exp(-(h**2)/(2*sigma**2))
        if args.Kernel == "Variable_Fraser_Suzuki":
            m = 0
            a = np.random.uniform(0.2+e, 0.6)
            sigma = np.random.uniform(0.25+e, 1)

            Nh = 15
            h = np.linspace(-Nh, Nh)

            for i in range(len(h)):
                if h[i] < m-(sigma/a):
                    h[i] = 0
                else:
                    pass
            T_f = Fraser_Suzuki(h[h != 0], a, m, sigma)
            T_f = list(h[h == 0])+list(T_f)
            h = T_f

        x_true, x_degraded, x_degraded_2000 = create_signal(
            h, molecule_number, m1, s1, args.Range)
        if np.max(x_true) > 30:
            print("num sig test", test_counter)
            H = convmtx(h, n_in)
            np.save(os.path.join(test_path_H, f'H_te_{test_counter}'), H)
            np.save(os.path.join(test_path_Groundtruth,
                    f'x_Gr_te_{test_counter}'), x_true.ravel())
            np.save(os.path.join(test_path_Degraded,
                    f'x_De_te_{test_counter}'), x_degraded.ravel())
            np.save(os.path.join(test_path_Degraded2000,
                    f'x_De_te_{test_counter}'), x_degraded_2000.ravel())

            test_counter += 1
