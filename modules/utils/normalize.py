from typing import Tuple
import numpy as np

class Normalize:
    """
    Normalizes the input 3D matrix with the given mean and standard deviation. 
    Performs per channel normalization of input matrix.

    Args
    ----

    mean (Tuple) : Tuple of means of Red Channel, Green Channel, Blue Channel
    std (Tuple) : Tuple of std of Red Channel, Green Channel, Blue Channel

    Returns
    -------
    Normalized `numpy.ndarray` with same shape as input 
    """

    def __init__(self,mean: Tuple, std: Tuple) -> None:
        self.mean = mean
        self.std = std

    def transform(self,matrix: np.ndarray) -> np.ndarray:
        
        shape = matrix.shape
        matrix = matrix/255.0
        matrix = matrix.transpose(2,0,1)
        r = matrix[0]
        g = matrix[1]
        b = matrix[2]

        r = (r-self.mean[0])/self.std[0]
        g = (g-self.mean[1])/self.std[1]
        b = (b-self.mean[2])/self.std[2]

        matrix[0] = r
        matrix[1] = g
        matrix[2] = b

        matrix = matrix.transpose(1,2,0)
        assert matrix.shape == shape
        return matrix
