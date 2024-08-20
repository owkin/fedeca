"""Module related to synthetic experiments."""
from __future__ import annotations

import itertools
import pickle
from typing import Any

import numpy as np
import pandas as pd


def load_dataframe_from_pickles(filename: str) -> pd.DataFrame:
    """Specialized function to load dataframe from a pickle file.

    Parameters
    ----------
    filename: str
        Path to the pickle file. Supposedly the output of
        `experiments.run_experiment` that contains a set of pickles, where each
        pickle is a list of pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        A single dataframe concatenating all results.
    """

    def load_pickles(filename: str):
        with open(filename, "rb") as file:
            while True:
                try:
                    yield pickle.load(file)
                except EOFError:
                    break
                except pickle.UnpicklingError:
                    continue

    return pd.concat(df for list_df in load_pickles(filename) for df in list_df)


def param_grid_from_dict(param_dict: dict[str, Any]) -> pd.DataFrame:
    """Generate a grid of parameters as pandas.DataFrame from a dictionary.

    Nested dictionary not supported.

    Returns
    -------
    pandas.DataFrame
        A dataframe where each column represents a parameter, each row represents
        a combination of parameters.
    """
    for key, value in param_dict.items():
        if pd.api.types.is_scalar(value):
            param_dict[key] = [value]
    return pd.DataFrame(
        itertools.product(*param_dict.values()),
        columns=list(param_dict.keys()),
    )


# def std_mean_differences(x, y):
#     """Compute standardized mean differences."""
#     print("WARNING this function cannot be used with weighted data !!!")
#     print("Use instead standardized_mean_diff in metrics/metrics.py")
#     std_x = np.std(x)
#     std_y = np.std(y)
#     if (std_x == 0) and (std_y == 0):
#         return np.mean(x) - np.mean(y)
#     return (np.mean(x) - np.mean(y)) / np.sqrt((std_x**2 + std_y**2) / 2)


def ratio_variances(x, y):
    """Compute ratio of variances."""
    std_x = np.std(x)
    std_y = np.std(y)
    if std_y == 0:
        return np.infty
    return std_x**2 / std_y**2


def effective_sample_size(w):
    """Compute effective sample size."""
    denom = np.sum(w**2)
    if denom > 0.0:
        return (np.sum(w) ** 2) / denom
    return 0
