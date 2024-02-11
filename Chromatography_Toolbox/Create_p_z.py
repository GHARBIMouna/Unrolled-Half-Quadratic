import numpy as np


def eval_sig_theoretical(x):
    """
    Returns for one given peak signal its associated 
    effective support, height and location for Groundtruth signals.
    """
    i = 0
    L_peak = []
    L_x_peak = []
    # The goal of this first loop is to isolate the peak individually
    while i < len(x):
        if x[i] != 0:
            j = i
            while x[j] != 0:
                L_peak.append(j)
                L_x_peak.append(x[j])
                j += 1
            i = j-1
        i += 1
    # The goal of this second part is to return H, W and A associated to isolated peak
    H = np.max(L_x_peak)
    L = list(x).index(np.max(x[L_peak]))
    for i in range(len(L_x_peak)-1):
        if H/20 > L_x_peak[i] and H/20 < L_x_peak[i+1]:
            Supp_eff_beg = i
        if H/20 < L_x_peak[i] and H/20 > L_x_peak[i+1]:
            Supp_eff_end = i+1
    Supp_eff = L_peak[Supp_eff_beg:Supp_eff_end]
    return Supp_eff, H, L


def x_creation(N, n, extrem_dist, peak_inter_dist, T_f):
    """
    Creates groundtruth signals starting by selecting n spikes,
    these spikes need to be unique, meaning no 
    two spikes are superposed, and the minimal distance between two spikes 
    is set by peak_inter_dist. Spikes intensities are then chosen and finally
    Spikes are then convolved with preselected kernel.
    """
    position_count = [0]*n
    position_diff = [0]*(n-1)

    # the first part of the condition ensures that no peaks are superposed
    # Second part of the condition ensure that the minimal distance between two
    # spikes is greater than a minimal set distance
    while position_count != [1]*n or np.min(position_diff) <= peak_inter_dist:
        # we ensure the peak position is between position extrem_dist and N-extrem_dist of the signal
        positions = np.random.randint(extrem_dist-1, N-extrem_dist, size=n)
        positions = np.sort(positions)
        position_count = [list(positions).count(i) for i in positions]
        position_rot = list(positions)[n-1:] + list(positions)[:n-1]
        position_diff = list(np.abs(positions-position_rot))[1:]
    peaks_inten = abs(np.random.randn(n))
    H_Gr = np.zeros(n,)
    L_Gr = np.zeros(n,)
    x = np.zeros((n, N))
    Supp_eff_mat = np.zeros((n, N))
    for i in range(n):
        # Seperate spikes into n different signals of size N
        x[i, positions[i]] = peaks_inten[i]
        # Convolve each spike with the chosen kernel
        x[i] = np.convolve(x[i], T_f, mode="same")
        Supp_eff, H, L = eval_sig_theoretical(x[i])
        Supp_eff_mat[i, Supp_eff] = np.ones(len(Supp_eff))
        H_Gr[i] = H
        L_Gr[i] = L
    # Sum the seperate spike signals to obtain the groundtruth signal
    x_true = np.sum(x, axis=0)
    return x_true, H_Gr, L_Gr, Supp_eff_mat


def y_degradation(x_true, m1, s1, T_h):
    """
    Creates a degraded signal associated to a groundtruth signal 
    x_true, by convolving with a blur kernel and adding a noise.
    """
    x_degraded = np.convolve(x_true, T_h)
    noise = np.random.normal(m1, s1, np.shape(x_degraded))
    x_degraded = x_degraded+noise
    return x_degraded
