import numpy as np


def rescale(x: np.array, a=0, b=1):
    '''Map values in range [a, b]'''
    x = np.array(x, dtype=float)
    x -= np.min(x)
    x /= np.max(x)
    return a + x * (b - a)
