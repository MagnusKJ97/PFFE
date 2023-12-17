import numpy as np
from typing import Union, Tuple
from scipy.interpolate import interp1d


def equal_weights(x: np.ndarray, axis: int = 0) -> np.ndarray:

    """
    Calculates equal weights

    Parameters
    ----------
    x:
    axis:

    Returns
    -------
    np.ndarray
    """

    n = x.shape[axis]

    return np.repeat(1.0 / n, n)