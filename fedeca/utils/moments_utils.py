"""A module containing utils to compute high-order moments using Newton's formeanla."""
from typing import Any, List

import numpy as np
import pandas as pd
from scipy.special import binom


def compute_uncentered_moment(data, order):
    """Compute the uncentered moment.

    Parameters
    ----------
    data : pd.DataFrame, np.array
        dataframe.
    order : int
        order of the moment.

    Returns
    -------
    pd.DataFrame, np.array
        Moment of order k.

    Raises
    ------
    NotImplementedError
        Raised if the data type is not Dataframe nor np.ndarray.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        moment = data.select_dtypes(include=np.number).pow(order).mean(skipna=True)
    elif isinstance(data, np.ndarray):
        moment = np.nanmean(np.power(data, order), axis=0)
    else:
        raise NotImplementedError(
            "Only DataFrame or numpy array are currently handled."
        )
    return moment


# pylint: disable=deprecated-typing-alias
def compute_centered_moment(uncentered_moments: List[Any]):
    r"""Compute the centered moment of order k.

    Given a list of the k first unnormalized moments,
    compute the centered moment of order k.
    For high values of the moments the results can
    differ from scipy.special.moment.
    We are interested in computing
    .. math::
        \hat{\mu}_k  = \frac{1}{\hat{\sigma}^k}
            \mathbb E_Z \left[ (Z - \hat{\mu})^k\right]
        \hat{\mu}_k  = \frac{1}{\hat{\sigma}^k}
            \mathbb E_Z \left[ \sum_{l=0}^k\binom{k}{l} Z^{k-l} (-1)^l\hat\mu^l)\right]
        \hat{\mu}_k  = \frac{1}{\hat{\sigma}^k}
          \sum_{l=0}^k(-1)^l\binom{k}{l} \mathbb E_Z \left[ Z^{k-l}
          \right]\mathbb E_Z \left[ Z \right]^l
    thus we only need the list uncentered moments up to order k.

    Parameters
    ----------
    uncentered_moments : List[Any]
        List of the k first non-centered moment.

    Returns
    -------
    Any
        The centered k-th moment.
    """
    mean = np.copy(uncentered_moments[0])
    order = len(uncentered_moments)
    result = (-mean) ** order  # i+1 = 0
    # We will go over the list of moments to add Newton's binomial
    # expansion formula terms one by one, where the current
    # moment is ahead of i by 1 hence we call it moment_i_plus_1
    for i, moment_i_plus_1 in enumerate(uncentered_moments):
        temp = (-mean) ** (order - i - 1)
        temp *= moment_i_plus_1  # the power is already computed
        temp *= binom(order, i + 1)
        result += temp
    return result


# pylint: disable=deprecated-typing-alias
def aggregation_mean(local_means: List[Any], n_local_samples: List[int]):
    """Aggregate local means.

    Aggregate the local means into a global mean by using the local number of samples.

    Parameters
    ----------
    local_means : List[Any]
        List of local means. Could be array, float, Series.
    n_local_samples : List[int]
        List of number of samples used for each local mean.

    Returns
    -------
    Any
        Aggregated mean. Same type of the local means
    """
    tot_samples = np.copy(n_local_samples[0])
    tot_mean = np.copy(local_means[0])
    for mean, n_sample in zip(local_means[1:], n_local_samples[1:]):
        mean = np.nan_to_num(mean, nan=0, copy=False)
        tot_mean *= tot_samples / (tot_samples + n_sample)
        tot_mean += mean * (n_sample / (tot_samples + n_sample))
        tot_samples += n_sample

    return tot_mean
