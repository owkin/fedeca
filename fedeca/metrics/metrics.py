"""Define metrics for ECA analysis."""

import numpy as np
import pandas as pd


def standardized_mean_diff(
    confounders, treated, weights=None, use_unweighted_variance=True
):
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
    use_unweighted_variance : bool
        if True, the variance is computed without weights. To follow
        https://stats.stackexchange.com/questions/618643/formula-for-standardized-mean-difference-in-cobalt-package-for-categorical-varia  # noqa: E501
        If False use recalibrated variance as in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4626409/.  # noqa: E501

    Returns
    -------
    smd: np.ndarray
        standardized mean differences of the confounders.
    """
    if weights is None:
        weights = np.ones((len(confounders.index)))

    if use_unweighted_variance:
        weights_var = np.ones((weights.shape[0]))
    else:
        weights_var = weights

    # unbiased var estimator see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4626409/  # noqa: E501
    # when use_unweighted_variance is True this just is n / (n - 1) for unbiasedness
    var_scaler_treated = weights_var[treated].sum() ** 2 / (
        weights_var[treated].sum() ** 2 - np.power(weights_var[treated], 2).sum()
    )
    var_scaler_untreated = weights_var[~treated].sum() ** 2 / (
        weights_var[~treated].sum() ** 2 - np.power(weights_var[~treated], 2).sum()
    )

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
                        - np.average(
                            confounders.loc[treated, cont_columns],
                            weights=weights_var[treated],
                            axis=0,
                        ),
                        2,
                    ),
                    weights=weights_var[treated],
                    axis=0,
                )
                + var_scaler_untreated
                * np.average(
                    np.power(
                        confounders.loc[~treated, cont_columns]
                        - np.average(
                            confounders.loc[~treated, cont_columns],
                            weights=weights_var[~treated],
                            axis=0,
                        ),
                        2,
                    ),
                    weights=weights_var[~treated],
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
                var_scaler_treated
                * treated_cat_confounders_avg
                * np.average(
                    1 - confounders.loc[treated, cat_columns],
                    weights=weights_var[treated],
                    axis=0,
                )
                + var_scaler_untreated
                * untreated_cat_confounders_avg
                * np.average(
                    1 - confounders.loc[~treated, cat_columns],
                    weights=weights_var[~treated],
                    axis=0,
                )
            )
            / 2
        ),
        index=cat_columns,
    )
    smd_cat *= 100

    smd = pd.concat([smd_continuous, smd_cat])
    smd = smd.reindex(confounders.columns)
    return smd
