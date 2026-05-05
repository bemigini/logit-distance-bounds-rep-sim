"""

Functions for constructing model embeddings and unembeddings 


"""




import numpy as np 
from numpy.typing import NDArray


def get_fx_from_rad_from_g_and_length(
    f_rad_from_g: NDArray, g_angle: NDArray, f_length: NDArray):
    """
    Get the embedding functions based on the angles of the unembedding vectors 
    and the radian from the correct unembedding 
    """
    f_angle = (g_angle + f_rad_from_g) % (2*np.pi)
    fx_a = f_length*np.cos(f_angle)
    fx_b = f_length*np.sin(f_angle)
    fx = np.concatenate((np.expand_dims(fx_a, 2), np.expand_dims(fx_b, 2)), 2)

    return fx 


def get_2dvectors_from_rad_and_length(
    rad: NDArray, length: NDArray|float):
    """ 
    Get the 2d vectors from the angles in radians
    and the lengths of the vectors. 
    """
    vec_a = length*np.cos(rad)
    vec_b = length*np.sin(rad)
    vectors = np.concatenate((np.expand_dims(vec_a, 2), np.expand_dims(vec_b, 2)), 2)

    return vectors
