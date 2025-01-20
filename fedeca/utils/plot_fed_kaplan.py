"""Plot the survival curve with confidence intervals."""
import matplotlib.pyplot as plt
import numpy as np
from lifelines.utils import inv_normal_cdf


def compute_ci(s, var_s, csum_var, alpha=0.05, ci="exp_greenwood"):
    """Compute the confidence interval of the survival function.

    Parameters
    ----------
    grid : np.ndarray
        The time grid to plot.
    s : np.ndarray
        The survival function computed at the grid.
    var_s : np.ndarray
        The variance of the survival function computed at the grid.
        Only needed if ci is "greenwood".
    csum_var : np.ndarray
        The cumulative sum of the variance of the survival function.
        Only needed if ci is "exp_greenwood".
    alpha : float, optional
        The significance level of the CI , by default 0.05
    ci : str, optional
        The method to draw CIs, by default "exp_greenwood"

    Returns
    -------
    np.ndarray
        The lower bound of the CI.
    np.ndarray
        The upper bound of the CI.
    """
    assert ci in [
        "exp_greenwood",
        "greenwood",
    ], "ci must be either 'exp_greenwood' or 'greenwood'"
    z = inv_normal_cdf(1 - alpha / 2)
    if ci == "exp_greenwood":

        # There will be -inf in the log when s is 0
        v = np.log(s)
        # we don't use directly V hat for more stable computation
        # (avoids sqrt(log(v)**2))
        lower = np.exp(-np.exp(np.log(-v) - z * np.sqrt(csum_var) / (v + 1e-16)))
        upper = np.exp(-np.exp(np.log(-v) + z * np.sqrt(csum_var) / (v + 1e-16)))
    elif ci == "greenwood":

        lower = s - z * np.sqrt(var_s)
        upper = s - z * np.sqrt(var_s)

    return lower, upper


def fed_km_plot(
    grid, s, var_s, csum_var, alpha=0.05, ci="exp_greenwood", label="KM_estimate", color=None,
):
    """Plot the survival curve with confidence intervals.

    Parameters
    ----------
    grid : np.ndarray
        The time grid to plot.
    s : np.ndarray
        The survival function computed at the grid.
    var_s : np.ndarray
        The variance of the survival function computed at the grid.
        Only needed if ci is "greenwood".
    csum_var : np.ndarray
        The cumulative sum of the variance of the survival function.
        Only needed if ci is "exp_greenwood".
    alpha : float, optional
        The significance level of the CI , by default 0.05
    ci : str, optional
        The method to draw CIs, by default "exp_greenwood"
    label : str, optional
        The label of the curve, by default "KM_estimate"
    color: Union[None, str]
        The color of the curve
    """
    lower, upper = compute_ci(s, var_s, csum_var, alpha=alpha, ci=ci)
    if color is None:
        ax = plt.plot(grid, s, label=label)
        plt.fill_between(grid, lower, upper, alpha=0.25, linewidth=1.0, step="post")
    else:
        ax = plt.plot(grid, s, label=label, color=color)
        plt.fill_between(grid, lower, upper, alpha=0.25, linewidth=1.0, step="post", color=color)
    plt.xlabel("timeline")
    return ax
