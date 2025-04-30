import numpy as np

class ColorShift:
    def __init__(self,r: int, g: int, b: int, p: float) -> None:
        self.r = r
        self.g = g
        self.b = b
        self.p = p

    def transform(self, image: np.ndarray) -> np.ndarray:
        color_shift_image = image
        if np.random.rand() >= self.p:
            R = image[:,:,0]
            G = image[:,:,1]
            B = image[:,:,2]
            color_shift_image = np.dstack( (
                np.roll(R, self.r, axis=0), 
                np.roll(G, self.g, axis=1), 
                np.roll(B, self.b, axis=0)
                ))
        return color_shift_image
