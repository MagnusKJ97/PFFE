import numpy as np
from scipy.linalg import block_diag
from sklearn.neighbors import KernelDensity
from typing import Tuple, Union

def cov_to_corr_matrix(cov_mat: np.ndarray) -> np.ndarray:

    """
    Transform a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov_mat:
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Correlation matrix.

    """

    vols = np.sqrt(np.diag(cov_mat))
    corr_mat = cov_mat / np.outer(vols, vols)
    corr_mat[corr_mat < -1], corr_mat[corr_mat > 1] = -1, 1  # numerical error

    return corr_mat


def corr_to_cov_matrix(corr_mat: np.ndarray, vols: np.ndarray) -> np.ndarray:

    """
    Transform a covariance matrix to a correlation matrix.

    Parameters
    ----------
    corr_mat:
        Correlation matrix.
    vols:
        Volatilities.

    Returns
    -------
    np.ndarray
        Covariance matrix.

    """

    cov_mat = corr_mat * np.outer(vols, vols)

    return cov_mat

def form_block_corr_matrix(num_blocks: int, block_size: int, block_corr: float) -> np.ndarray:

    """
    Create a block correlation matrix with a number of equal size blocks.
    Each block have the same inter block correlation.

    Parameters
    ----------
    num_blocks:
        Number of blocks
    block_size:
        Block size.
    block_corr
        Inter block correlation.

    Returns
    -------
    np.ndarray
        Correlation matrix.

    """

    block = np.ones((block_size, block_size)) * block_corr
    np.fill_diagonal(block, 1.0)

    corr_mat = block_diag(*([block] * num_blocks))

    return corr_mat

