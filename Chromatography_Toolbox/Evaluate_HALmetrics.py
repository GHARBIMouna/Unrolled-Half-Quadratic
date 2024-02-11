import os
import numpy as np
import matplotlib.pyplot as plt


def eval_air(x_true, Supp_eff_mat_Gr):
    """returns the groundtruth area of peaks on their effective support"""
    X_true = x_true*Supp_eff_mat_Gr
    A_true = np.zeros(np.shape(X_true)[0],)
    for i in range(np.shape(X_true)[0]):
        for j in range(np.shape(X_true)[1]-1):
            A_true[i] += (X_true[i][j]+X_true[i][j+1])/2
    return A_true


def eval_reconstruction(x_pred, Supp_eff_mat_Gr):
    """
    Returns vectors of predicted peak intensity H_pred,
    peak position L_pred and under curve area A_pred for
    an input signal.
    """

    X_pred = x_pred*Supp_eff_mat_Gr
    H_pred = np.max(X_pred, axis=1)
    L_pred = np.argmax(X_pred, axis=1)
    A_pred = np.zeros(np.shape(X_pred)[0],)
    for i in range(np.shape(X_pred)[0]):
        if L_pred[i] == 0:
            # Peak predicted to zero then choose a location randomly on the support
            L_pred[i] = np.random.randint(np.nonzero(Supp_eff_mat_Gr[i])[
                                          0][0], np.nonzero(Supp_eff_mat_Gr[i])[0][-1]+1)
        for j in range(np.shape(X_pred)[1]-1):
            A_pred[i] += (X_pred[i][j]+X_pred[i][j+1])/2
    return H_pred, L_pred, A_pred


def SUPP_Overlap(Supp_eff_mat_Gr, H_pred, A_pred, L_pred, H_Gr, A_Gr, L_Gr):
    H_pred_LHalf = []
    L_pred_LHalf = []
    A_pred_LHalf = []

    H_Gr_LHalf = []
    L_Gr_LHalf = []
    A_Gr_LHalf = []

    H_pred_MHalf = []
    A_pred_MHalf = []
    L_pred_MHalf = []

    H_Gr_MHalf = []
    A_Gr_MHalf = []
    L_Gr_MHalf = []
    i = 0
    thresh = 0.3
    while (i < (np.shape(Supp_eff_mat_Gr)[0]-1)):

        if i != np.shape(Supp_eff_mat_Gr)[0]-1:
            if len(np.intersect1d(np.nonzero(Supp_eff_mat_Gr[i])[0], np.nonzero(Supp_eff_mat_Gr[i+1])[0])) < thresh*len(np.nonzero(Supp_eff_mat_Gr[i])[0]):
                H_pred_LHalf.append(H_pred[i])
                A_pred_LHalf.append(A_pred[i])
                L_pred_LHalf.append(L_pred[i])

                H_Gr_LHalf.append(H_Gr[i])
                A_Gr_LHalf.append(A_Gr[i])
                L_Gr_LHalf.append(L_Gr[i])
                i += 1
        if i != np.shape(Supp_eff_mat_Gr)[0]-1:
            if len(np.intersect1d(np.nonzero(Supp_eff_mat_Gr[i])[0], np.nonzero(Supp_eff_mat_Gr[i+1])[0])) >= thresh*len(np.nonzero(Supp_eff_mat_Gr[i])[0]):
                H_pred_MHalf.append(H_pred[i])
                A_pred_MHalf.append(A_pred[i])
                L_pred_MHalf.append(L_pred[i])

                H_Gr_MHalf.append(H_Gr[i])
                A_Gr_MHalf.append(A_Gr[i])
                L_Gr_MHalf.append(L_Gr[i])

                H_pred_MHalf.append(H_pred[i+1])
                A_pred_MHalf.append(A_pred[i+1])
                L_pred_MHalf.append(L_pred[i+1])

                H_Gr_MHalf.append(H_Gr[i+1])
                A_Gr_MHalf.append(A_Gr[i+1])
                L_Gr_MHalf.append(L_Gr[i+1])
                i += 2

    if (len(H_pred)-len(H_pred_LHalf)-len(H_pred_MHalf)) == 1:

        H_pred_LHalf.append(H_pred[np.shape(Supp_eff_mat_Gr)[0]-1])
        A_pred_LHalf.append(A_pred[np.shape(Supp_eff_mat_Gr)[0]-1])
        L_pred_LHalf.append(L_pred[np.shape(Supp_eff_mat_Gr)[0]-1])

        H_Gr_LHalf.append(H_Gr[np.shape(Supp_eff_mat_Gr)[0]-1])
        A_Gr_LHalf.append(A_Gr[np.shape(Supp_eff_mat_Gr)[0]-1])
        L_Gr_LHalf.append(L_Gr[np.shape(Supp_eff_mat_Gr)[0]-1])
    return H_pred_LHalf, H_pred_MHalf,  L_pred_LHalf, L_pred_MHalf,  A_pred_LHalf, A_pred_MHalf,  H_Gr_LHalf, H_Gr_MHalf,  L_Gr_LHalf, L_Gr_MHalf,  A_Gr_LHalf, A_Gr_MHalf


def eval_rec_mthd(path_pred, path, num_signals, mode):
    """Evaluates the performance of a reconstruction method using HALmetrics 
    returns the $\ell_1$ norm of the differences in peaks heights H, locations L,
    and area under curve A.    
    """
    H_fig = plt.figure(figsize=(3, 3), dpi=300)
    L_fig = plt.figure(figsize=(3, 3), dpi=300)
    A_fig = plt.figure(figsize=(3, 3), dpi=300)
    H_Avg_l1 = 0
    A_Avg_l1 = 0
    L_Avg_l1 = 0
    H_l1_list = []
    L_l1_list = []
    A_l1_list = []
    for i in range(num_signals):
        x_true = np.load(os.path.join(
            path, mode, "Groundtruth", "x_Gr_te" + "_"+str(i)+".npy"))
        x_pred = np.load(os.path.join(
            path_pred, "Rec_Signals", "x_Es_te" + "_"+str(i)+".npy"))
        H_Gr = np.load(os.path.join(path, mode, "Infos", "H",
                       "x_In_H_"+mode[0:2]+"_"+str(i)+".npy"))
        L_Gr = np.load(os.path.join(path, mode, "Infos", "L",
                       "x_In_L_"+mode[0:2]+"_"+str(i)+".npy"))
        Supp_eff_mat_Gr = np.load(os.path.join(
            path, mode, "Infos", "Supp-Eff", "x_In_SE_"+mode[0:2]+"_"+str(i)+".npy"))
        A_Gr_computed = eval_air(x_true, Supp_eff_mat_Gr)
        H_pred, L_pred, A_pred = eval_reconstruction(x_pred, Supp_eff_mat_Gr)

        # Original chemical metrics
        H_l1 = abs(H_pred-H_Gr).sum()
        A_l1 = abs(A_Gr_computed-A_pred).sum()
        L_l1 = abs(L_pred-L_Gr).sum()

        # Compute normalized chemical metrics
        H_l1 = H_l1/(abs(H_Gr).sum())
        A_l1 = A_l1/(abs(A_Gr_computed).sum())
        L_l1 = H_l1/(abs(L_Gr).sum())

        H_l1_list.append(H_l1)
        L_l1_list.append(L_l1)
        A_l1_list.append(A_l1)

        H_Avg_l1 += H_l1
        A_Avg_l1 += A_l1
        L_Avg_l1 += L_l1

        H_pred_LHalf, H_pred_MHalf, L_pred_LHalf, L_pred_MHalf, A_pred_LHalf, A_pred_MHalf, H_Gr_LHalf, H_Gr_MHalf,  L_Gr_LHalf, L_Gr_MHalf,  A_Gr_LHalf, A_Gr_MHalf = SUPP_Overlap(
            Supp_eff_mat_Gr, H_pred, A_pred, L_pred, H_Gr, A_Gr_computed, L_Gr)
        #### Scatter plots####
        plt.figure(H_fig.number)
        plt.scatter(H_Gr_LHalf, H_pred_LHalf, s=2, marker="o", color="c")
        plt.scatter(H_Gr_MHalf, H_pred_MHalf, s=2, marker="v", color="y")
        plt.xlim([-0.1, 4])
        plt.ylim([-0.1, 4])
        plt.xticks([0, 1, 2, 3, 4])
        plt.yticks([0, 1, 2, 3, 4])
        plt.xlabel(r"$\overline{H}$")
        plt.ylabel(r"$\widehat{H}$")

        plt.figure(A_fig.number)
        plt.scatter(A_Gr_LHalf, A_pred_LHalf, s=2, marker="o", color="c")
        plt.scatter(A_Gr_MHalf, A_pred_MHalf, s=2, marker="v", color="y")
        plt.xlim([-0.5, 30])
        plt.ylim([-0.5, 30])
        plt.xticks([0, 10, 20, 30])
        plt.yticks([0, 10, 20, 30])
        plt.xlabel(r"$\overline{A}$")
        plt.ylabel(r"$\widehat{A}$")

        plt.figure(L_fig.number)
        plt.scatter(L_pred_LHalf, list(np.array(L_Gr_LHalf) -
                    np.array(L_pred_LHalf)), s=2, marker="o", color="c")
        plt.scatter(L_pred_MHalf, list(np.array(L_Gr_MHalf) -
                    np.array(L_pred_MHalf)), s=2, marker="v", color="y")
        plt.xticks([0, 1000, 2000])
        plt.xlabel(r"$\overline{L}$")
        plt.ylabel(r"$\widehat{L}-\overline{L}$")

    # SAVING Scatter Plots
    plt.figure(H_fig.number)
    plt.grid()
    plt.savefig(os.path.join(path_pred, "H_scatter"),
                dpi=200, bbox_inches="tight")

    plt.figure(A_fig.number)
    plt.grid()
    plt.savefig(os.path.join(path_pred, "A_scatter"),
                dpi=200, bbox_inches="tight")

    plt.figure(L_fig.number)
    plt.grid()
    plt.savefig(os.path.join(path_pred, "L_scatter"),
                dpi=200, bbox_inches="tight")

    #### METRICS AVERAGE ####
    H_Avg_l1 = H_Avg_l1/num_signals
    A_Avg_l1 = A_Avg_l1/num_signals
    L_Avg_l1 = L_Avg_l1/num_signals
    ########################

    #### METRICS STD ####
    H_std_l1 = 0
    for h in H_l1_list:
        H_std_l1 = H_std_l1+((h-H_Avg_l1)**2)
    H_std_l1 = np.sqrt(H_std_l1/(num_signals-1))

    A_std_l1 = 0
    for a in A_l1_list:
        A_std_l1 = A_std_l1+((a-A_Avg_l1)**2)
    A_std_l1 = np.sqrt(A_std_l1/(num_signals-1))

    L_std_l1 = 0
    for l in L_l1_list:
        L_std_l1 = L_std_l1+((l-L_Avg_l1)**2)
    L_std_l1 = np.sqrt(L_std_l1/(num_signals-1))
    #####################
    return H_Avg_l1, A_Avg_l1, L_Avg_l1, H_std_l1, A_std_l1, L_std_l1


# Give a path of a dataset, e.g.,
path = './Chromatography_Toolbox/Dataset1/'

# Give a path of a learnt model, where estimated signals using a method were saved in a folder named "Rec_Signals". e.g., U_ISTA
path_pred = './Chromatography_Toolbox/Dataset1//Trained_Model_ISTA_1/'

num_signals = 10
mode = 'test'
H_Avg_l1, A_Avg_l1, L_Avg_l1, H_std_l1, A_std_l1, L_std_l1 = eval_rec_mthd(
    path_pred, path, num_signals, mode)


print("Average l1 height H difference: ", H_Avg_l1)
print("L1 height H difference STD: ", H_std_l1)
print("Average l1 Aire A difference: ", A_Avg_l1)
print("L1 Aire A difference STD: ", A_std_l1)
print("Average l1 location L difference: ", L_Avg_l1)
print("L1 location L difference STD: ", L_std_l1)
