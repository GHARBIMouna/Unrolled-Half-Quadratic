import numpy as np


def Fraser_Suzuki(t, a, m, sigma):
    """
    Creates a Frazer Suzuki shaped kernel.

    Parameters:
    ----------
    t:     list of input values
    a:     peak tailing
    m:     mean
    sigma: peak width
    """
    return 1/(sigma * np.sqrt(2 * np.pi)*np.exp((np.log(2)*a**2)/4))*np.exp((-1/(2*a**2)) * np.log(1 + a*(t-m)/sigma)**2)


def Gaussian(t, m, s):
    """
    Creates a Gaussian shaped kernel.

    Parameters:
    ----------
    t: list of input values
    m: mean
    s: standard deviation
    """
    return 1/(s * np.sqrt(2 * np.pi)) * np.exp(-(t-m)**2 / (2 * s**2))


def convmtx(h, n_in):
    """
    Generates a convolution matrix H such that the product
    of H and a vector x of size n_in corresponds to the 
    convolution of H and x.
    Usage: H = convm(h,n_in)
    Given a column vector h of length N, an (N+n_in-1 x n_in)_in convolution matrix is
    generated
    This method has the same functionning as that of np.convolve(x,h)

    Paramters:
    ---------
    h: vector of kernel
    n_in: size of vector x, 2nd dimension of matrix H.
    """
    N = len(h)
    N1 = N + 2*n_in - 2
    hpad = np.concatenate([np.zeros(n_in-1), h[:], np.zeros(n_in-1)])

    H = np.zeros((len(h)+n_in-1, n_in))

    for i in range(n_in):
        H[:, i] = hpad[n_in-i-1:N1-i]
    return H
