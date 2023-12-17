import numpy as np
from scipy import stats
from typing import Tuple, Union, List
from codelibstatisticshistoricalprobabilities import equal_weights
from codelibportfoliooptimizationriskmetrics import calculate_conditional_value_at_risk, calculate_value_at_risk


def calculate_marginal_risks_std(weights: np.ndarray, cov_mat: np.ndarray) -> np.ndarray:
    """
    Function that calculates marginal risk using std. as portfolio risk measure
    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    total_risk = np.sqrt(weights @ cov_mat @ weights)
    inner_derivative = cov_mat @ weights

    return inner_derivative / total_risk


def calculate_risk_contributions_std(weights: np.ndarray, cov_mat: np.ndarray, scale: bool = False) -> np.ndarray:
    """
    Function that calculates risk contributions using std. as portfolio risk measure

    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix
    scale:
        Scale risk contribution.

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    mr = calculate_marginal_risks_std(weights, cov_mat)
    mrc = weights * mr

    if scale:
        mrc /= np.sum(mrc)

    return mrc


def calculate_marginal_sharpe(weights: np.ndarray, cov_mat: np.ndarray, mu: np.ndarray, rf: float):

    """
    Function that calculates marginal Sharpe ratio

    Parameters
    ----------
    weights:
        Portfolio weights
    cov_mat:
        Covariance matrix
    mu:
        Expected return vector.
    rf:
        Risk free rate.

    Returns
    -------
    np.ndarray
        Marginal risks
    """

    mr = calculate_marginal_risks_std(weights, cov_mat)
    excess_mu = mu - rf

    return excess_mu / mr