"""Define metrics for ECA analysis."""

import numpy as np
import pandas as pd


def standardized_mean_diff(confounders, treated, weights=None):
    """Compute the Standardized Mean Differences (SMD).

    Compute the Standardized Mean Differences between
    treated and control patients.

    Parameters
    ----------
    confounders : pd.DataFrame
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

    # unbiased var estimater
    var_scaler_treated = weights[treated].sum() / (weights[treated].sum() - 1)
    var_scaler_untreated = weights[~treated].sum() / (weights[~treated].sum() - 1)

    n_unique = confounders.nunique()
    cat_variables = n_unique == 2
    continuous_variables = n_unique != 2

    cont_columns = confounders.columns[continuous_variables].tolist()
    cat_columns = confounders.columns[cat_variables].tolist()

    treated_confounders_avg = np.average(
        confounders.loc[treated, cont_columns], weights=weights[treated], axis=0
    )
    untreated_confounders_avg = np.average(
        confounders.loc[~treated, cont_columns],
        weights=weights[~treated],
        axis=0,
    )
    # Returning back to a Series object as would .mean() do
    smd_continuous = pd.Series(
        treated_confounders_avg - untreated_confounders_avg, index=cont_columns
    )
    smd_continuous /= pd.Series(
        np.sqrt(
            (
                var_scaler_treated
                * np.average(
                    np.power(
                        confounders.loc[treated, cont_columns]
                        - treated_confounders_avg,
                        2,
                    ),
                    weights=weights[treated],
                    axis=0,
                )
                + var_scaler_untreated
                * np.average(
                    np.power(
                        confounders.loc[~treated, cont_columns]
                        - untreated_confounders_avg,
                        2,
                    ),
                    weights=weights[~treated],
                    axis=0,
                )
            )
            / 2
        ),
        index=cont_columns,
    )

    smd_continuous *= 100

    treated_cat_confounders_avg = np.average(
        confounders.loc[treated, cat_columns], weights=weights[treated], axis=0
    )
    untreated_cat_confounders_avg = np.average(
        confounders.loc[~treated, cat_columns], weights=weights[~treated], axis=0
    )

    smd_cat = pd.Series(
        treated_cat_confounders_avg - untreated_cat_confounders_avg, index=cat_columns
    )
    # TODO check if scaling of variance is correct
    smd_cat /= pd.Series(
        np.sqrt(
            (
                treated_cat_confounders_avg
                * np.average(
                    1 - confounders.loc[treated, cat_columns],
                    weights=weights[treated],
                    axis=0,
                )
                + untreated_cat_confounders_avg
                * np.average(
                    1 - confounders.loc[~treated, cat_columns],
                    weights=weights[~treated],
                    axis=0,
                )
            )
            / 2
        ),
        index=cat_columns,
    )
    smd_cat *= 100

    smd = pd.concat([smd_continuous, smd_cat])
    return smd
