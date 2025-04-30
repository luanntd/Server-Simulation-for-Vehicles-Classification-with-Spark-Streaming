import numpy as np

class RandomHorizontalFlip:
    def __init__(self, p:float) -> None:
        self.p = p

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = matrix.transpose(2,0,1)
        if np.random.rand() >= self.p:
            matrix = matrix[:,:,::-1]
        return matrix.transpose(1,2,0)

class RandomVerticalFlip:
    def __init__(self, p:float) -> None:
        self.p = p

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = matrix.transpose(2,0,1)
        if np.random.rand() >= self.p:
            matrix = matrix[:,::-1]
        return matrix.transpose(1,2,0)