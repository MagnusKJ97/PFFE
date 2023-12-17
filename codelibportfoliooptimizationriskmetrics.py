import numpy as np
from scipy import stats
from codelibstatisticsmoments import weighted_percentile
from codelibstatisticshistoricalprobabilities import equal_weights

from typing import Union, List

def calculate_value_at_risk(x: np.ndarray, p: Union[float, List[float]], probs: Union[np.ndarray, None] = None,
                            axis=0) -> Union[float, List]:

    """
    Calculates the Value-at-Risk.

    Parameters
    ----------
    x:
        Matrix of values.
    p:
        Value-at-risk level.
    probs:
        Weights for calculation of weighted Value-at-Risk. Default is equally weighted.
    axis:
        Axis to calculate value at risk over.

    Returns
    -------
        Value-At-Risk for a given level or list with Value-at-Risk for different levels.

    """

    return weighted_percentile(x, p, probs=probs, axis=axis)


def calculate_conditional_value_at_risk(x: np.ndarray, p: float, probs: np.ndarray = None, axis: int = 0) -> Union[float, List]:

    """
    Calculates the Conditional Value-at-Risk.

    Parameters
    ----------

    x:
        Matrix of values.
    p:
        Conditional Value-at-risk level.
    probs:
        Weights for calculation of weighted Conditional Value-at-Risk. Default is equally weighted.
    axis:
        Axis to calculate cond. value at risk over.

    Returns
    -------
    Union[float, List]
        Cond. Value-At-Risk for a given level or list with Cond. Value-at-Risk for different levels.

    """

    assert (p >= 0.0) and (p <= 1.0), 'Percentile must be between 0.0 and 1.0'

    sorted_idx = np.argsort(x, axis=axis)
    sorted_values = np.take_along_axis(x, sorted_idx, axis=axis)
    sorted_probs = equal_weights(x, axis=axis) if probs is None else probs[sorted_idx]

    if x.ndim == 1:
        value_at_risk_matrix = np.repeat(calculate_value_at_risk(x, p, probs, axis), x.shape[axis])
    else:
        value_at_risk_matrix = np.tile(calculate_value_at_risk(x, p, probs, axis), (x.shape[axis], 1))
        sorted_probs = np.tile(sorted_probs, (x.shape[1], 1)).T if axis == 0 else np.tile(sorted_probs, (x.shape[0], 1))

    if axis == 1:
        tail_bool = sorted_values <= np.transpose(value_at_risk_matrix)
    else:
        tail_bool = sorted_values <= value_at_risk_matrix

    unscaled_tail_probs = sorted_probs
    unscaled_tail_probs[~tail_bool] = 0

    if x.ndim == 1:
        scale_denom = np.sum(unscaled_tail_probs, axis=axis)
    else:
        scale_denom = np.atleast_2d(np.sum(unscaled_tail_probs, axis=axis))

    if axis == 1:
        scale_denom = scale_denom.T
    elif axis != 0:
        raise IndexError("Axis parameter is invalid, must be either 0 or 1")

    scaled_tail_probs = unscaled_tail_probs / scale_denom
    result = np.average(sorted_values, weights=scaled_tail_probs, axis=axis)

    return result
