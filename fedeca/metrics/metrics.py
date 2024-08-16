"""Define metrics for ECA analysis."""

import numpy as np
import pandas as pd


def standardized_mean_diff(confounders, treated, weights=None):
    """Compute the Standardized Mean Differences (SMD).

    Compute the Standardized Mean Differences between
    treated and control patients.

    Parameters
    ----------
    confounders : np.ndarray
        confounders array.
    treated : np.ndarray
        mask of booleans giving information about treated patients.
    weights : np.ndarray
        weights for the aggregation

    Returns
    -------
    smd: np.ndarray
        standardized mean differences of the confounders.
    """
    if weights is None:
        weights = np.ones_like(confounders)

    n_unique = confounders.nunique()
    cat_variables = n_unique == 2
    continuous_variables = n_unique != 2

    treated_confounders_avg = np.average(
        confounders.loc[treated, continuous_variables], weights=weights[treated], axis=0
    )
    untreated_confounders_avg = np.average(
        confounders.loc[~treated, continuous_variables],
        weights=weights[~treated],
        axis=0,
    )

    smd_continuous = treated_confounders_avg - untreated_confounders_avg
    smd_continuous /= np.sqrt(
        (
            np.average(
                np.power(
                    confounders.loc[treated, continuous_variables]
                    - treated_confounders_avg,
                    order=2,
                ),
                weights=weights[treated],
                axis=0,
            )
            + np.average(
                np.power(
                    confounders.loc[~treated, continuous_variables]
                    - untreated_confounders_avg,
                    order=2,
                ),
                weights=weights[~treated],
                axis=0,
            )
        )
        / 2
    )
    smd_continuous *= 100

    treated_cat_confounders_avg = np.average(
        confounders.loc[treated, cat_variables], weights=weights[treated], axis=0
    )
    untreated_cat_confounders_avg = np.average(
        confounders.loc[~treated, cat_variables], weights=weights[~treated], axis=0
    )

    smd_cat = treated_cat_confounders_avg - untreated_cat_confounders_avg
    smd_cat /= np.sqrt(
        (
            treated_cat_confounders_avg
            * np.average(
                1 - confounders.loc[treated, cat_variables],
                weights=weights[treated],
                axis=0,
            )
            + untreated_cat_confounders_avg
            * np.average(
                1 - confounders.loc[~treated, cat_variables],
                weights=weights[~treated],
                axis=0,
            )
        )
        / 2
    )
    smd_cat *= 100

    smd = pd.concat([smd_continuous, smd_cat])
    return smd
