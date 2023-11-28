"""Define metrics for ECA analysis."""

import numpy as np
import pandas as pd


def standardized_mean_diff(confounders, treated):
    """Compute the Standardized Mean Differences (SMD).

    Compute the Standardized Mean Differences between
    treated and control patients.

    Parameters
    ----------
    confounders : np.ndarray
        confounders array.
    treated : np.ndarray
        mask of booleans giving information about treated patients.

    Returns
    -------
    smd: np.ndarray
        standardized mean differences of the confounders.
    """
    n_unique = confounders.nunique()
    cat_variables = n_unique == 2
    continuous_variables = n_unique != 2

    smd_continuous = (
        confounders.loc[treated, continuous_variables].mean()
        - confounders.loc[~treated, continuous_variables].mean()
    )
    smd_continuous /= np.sqrt(
        (
            confounders.loc[treated, continuous_variables].var()
            + confounders.loc[~treated, continuous_variables].var()
        )
        / 2
    )
    smd_continuous *= 100

    smd_cat = (
        confounders.loc[treated, cat_variables].mean()
        - confounders.loc[~treated, cat_variables].mean()
    )
    smd_cat /= np.sqrt(
        (
            confounders.loc[treated, cat_variables].mean()
            * (1 - confounders.loc[treated, cat_variables]).mean()
            + confounders.loc[~treated, cat_variables].mean()
            * (1 - confounders.loc[~treated, cat_variables]).mean()
        )
        / 2
    )
    smd_cat *= 100

    smd = pd.concat([smd_continuous, smd_cat])
    return smd
