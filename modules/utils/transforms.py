from typing import List
import numpy as np

class Transforms:
    def __init__(self, transforms : List) -> None:
        self.transforms = transforms

    def transform(self, input : np.ndarray) -> np.ndarray:
        for transformer in self.transforms:
            input = transformer.transform(input)

        return input