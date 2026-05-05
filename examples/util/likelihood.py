"""

Functions for calculating log-likelihoods from embeddings and unembeddings


"""


import numpy as np 
from numpy.typing import NDArray

from scipy.special import logsumexp



def dot(fx: NDArray, gy: NDArray):
    """ Dot product numpy """
    return np.matmul(np.expand_dims(fx, 1), np.expand_dims(gy, 2))


def log_likelihood(fx: NDArray, gy: NDArray, all_gys: NDArray):
    """ Calculate log-likelihood """
    fg_dot = dot(fx, gy)

    normalisation = np.zeros((*fg_dot.shape, all_gys.shape[0]))

    for i, current_target in enumerate(all_gys):
        n = dot(fx, current_target)
        normalisation[:, :, :, i] = n 
    
    normalisation = logsumexp(normalisation, axis = 3, keepdims = False)

    return fg_dot - normalisation
