"""

For calculating the mean canonical correlation m_CCA

"""


import torch
import numpy as np 
from numpy.typing import NDArray

from sklearn.cross_decomposition import CCA



def mCCA(rep1: NDArray, rep2: NDArray, n_components: int) -> float:
    """ Get mean canonical correlation of the two representations.
        Uses [n_components] components for the CCA.
    """
    cca = CCA(n_components=n_components, max_iter=1000)
    cca.fit(rep1, rep2)

    # Mean of the CCA correlations
    X_c, Y_c = cca.transform(rep1, rep2)
    corrs = [np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)]
    mean_corr = np.mean(corrs)

    return mean_corr, cca.x_rotations_, cca.y_rotations_, np.max(corrs) 


def mCCA_tensors(rep1: torch.Tensor, rep2: torch.Tensor, n_components: int) -> float:
    """ Get mean canonical correlation of the two representations given as tensors.
        Uses [n_components] components for the CCA.
    """
    rep1 = rep1.detach().cpu().numpy() 
    rep2 = rep2.detach().cpu().numpy() 

    return mCCA(rep1, rep2, n_components)
