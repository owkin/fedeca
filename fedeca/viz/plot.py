"""File for experiment figure plots."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.axes import Axes

SIGNIFICANCE_THRESHOLD = 0.05

# create custom color palette
owkin_palette = {
    "owkin_pink": "#FFBDE0",
    "owkin_teal": "#32C6C6",
    "owkin_mustard": "#FFC200",
    "owkin_blue": "#1439C1",
    "owkin_magenta": "#F70B9D",
    "owkin_bright_blue": "#009EFF",
    "owkin_green": "#2CB546",
    "owkin_stone": "#D6CDC7",
    "owkin_black": "#000000",
}


def setup_owkin_colors_palette():
    """Add custom Owkin colors to mcolors."""
    owkin_palette_pd = pd.DataFrame(
        {"name": owkin_palette.keys(), "color": owkin_palette.values()}
    )

    c = dict(zip(*owkin_palette_pd.to_numpy().T))
    mcolors.get_named_colors_mapping().update(c)


def plot_power(
    df_res: pd.DataFrame,
    fit_curve: bool = False,
    deg: int = 2,
    plot_kwargs: dict | None = None,
) -> Axes:
    """Plot power or type I error figure for given experiment.

    Parameters
    ----------
    event_col : str, optional
        Column name for event indicator, by default "event".
    df_res : pd.DataFrame
        Results of experiment containing in every row the results
        for an experiment configuration and a column "p" with the p-value.
    fit_curve : bool, optional
        Interpolate the datapoints. Defaults to False.
    deg : int, optional
        Degree of polynomial for interpolation. Defaults to 2.
    plot_kwargs : Optional[dict], optional
        Parameter to plot on the xaxis if different from n_samples. Defaults to None.

    Returns
    -------
    matplotlib.axes._axes.Axes
        Power or type I error plot
    """
    setup_owkin_colors_palette()
    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs.setdefault("xlabel", "n_samples")
    param_vary = plot_kwargs["xlabel"]
    axis = plot_kwargs.pop("ax", plt.gca())

    df_power = (
        df_res.groupby(["method", param_vary])
        .agg(
            power=pd.NamedAgg(
                column="p",
                aggfunc=lambda x: (x < SIGNIFICANCE_THRESHOLD).sum() / x.size,
            ),
        )
        .reset_index()
    )
    owkin_colors = itertools.cycle(owkin_palette.keys())
    markers = ["d", "v", "s", "^", "*"]
    markers = itertools.cycle(markers)
    for name, group in df_power.groupby("method"):
        if (xlabel := plot_kwargs.pop("xlabel", None)) is not None:
            axis.set_xlabel(xlabel)
        if (ylabel := plot_kwargs.pop("ylabel", None)) is not None:
            axis.set_ylabel(ylabel)
        owkin_color = next(owkin_colors)
        marker = next(markers)
        if fit_curve:
            fit = np.poly1d(np.polyfit(group[param_vary], group["power"], deg=deg))
            axis.plot(group[param_vary], fit(group[param_vary]), color=owkin_color)
        axis.scatter(
            param_vary,
            "power",
            data=group,
            label=name,
            color=owkin_color,
            marker=marker,
            **plot_kwargs,
        )
        axis.legend()

    return axis


def plot_survival_function(
    data: pd.DataFrame,
    groupby: str | Sequence[str] | None = None,
    col_time: str = "time",
    col_sf: str = "sf",
    col_sf_lower: str | None = "sf_lower",
    col_sf_upper: str | None = "sf_upper",
    ax_set_kwargs: Mapping | None = None,
    **kwargs,
):
    """Create survival function plot based on time survival probability data.

    Parameters
    ----------
    data : dataframe
        Dataframe containing time points, survival probabilities, and optionally
        the upper and lower bounds of survival probabilities.
    groupby : str or Sequence of str, default=None
        Column name(s) used to group the data. Each group of data will be used
        to create a survival curve in the result.
    col_time : str, default="time"
        Name of the column of timepoints.
    col_sf : str, default="sf"
        Name of the column of survival probabilities.
    col_sf_lower : str, optional, default="sf_lower"
        Name of the column of the lower bounds of the survival probabilities
        used to draw the confidence interval band. If not found in the data
        or explicitly set to `None`, the band will not be drawn.
    col_sf_upper : str, optional, default="sf_upper"
        Name of the column of the lower bounds of the survival probabilities
        used to draw the confidence interval band. If not found in the data
        or explicitly set to `None`, the band will not be drawn.
    ax_set_kwargs : Mapping, optional
        Optional key value mapping to be given to the `set` function of the axis
        of the plot.
    **kwargs
        Extra arguments for seaborn's `lineplot`.
    """
    ax = kwargs.pop("ax", plt.gca())
    if ax_set_kwargs is None:
        ax_set_kwargs = {}

    # Due to numerical precision issue, bounds may be incorrect when the
    # survival probability is too small.
    if col_sf_lower in data and col_sf_upper in data:
        out_of_bounds = data[col_sf_upper].lt(data[col_sf]) | data[col_sf_lower].gt(
            data[col_sf]
        )
        data.loc[out_of_bounds, col_sf_upper] = data.loc[out_of_bounds, col_sf]
        data.loc[out_of_bounds, col_sf_lower] = data.loc[out_of_bounds, col_sf]

    if groupby is None:
        grouped = ((kwargs.get("label", "KM_estimate"), data),)
    else:
        grouped = data.groupby(groupby)
    kwargs.pop("label", None)

    for name, df in grouped:
        sns.lineplot(
            df,
            x=col_time,
            y=col_sf,
            drawstyle="steps-pre",
            ax=ax,
            label=name,
            **kwargs,
        )
        if col_sf_lower in data and col_sf_upper in data:
            ax.fill_between(
                x=col_time,
                y1=col_sf_lower,
                y2=col_sf_upper,
                alpha=0.25,
                step="pre",
                data=df,
                color=ax.lines[-1].get_color(),
            )
    ax.set(**ax_set_kwargs)

    return ax


if __name__ == "__main__":
    setup_owkin_colors_palette()
