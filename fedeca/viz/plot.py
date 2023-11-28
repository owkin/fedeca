"""File for experiment figure plots."""
import itertools
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.axes import Axes

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

    c = dict(zip(*owkin_palette_pd.values.T))
    mcolors.get_named_colors_mapping().update(c)


def plot_power(
    df_res: pd.DataFrame,
    fit_curve: bool = False,
    deg: int = 2,
    plot_kwargs: Optional[dict] = None,
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
            power=pd.NamedAgg(column="p", aggfunc=lambda x: (x < 0.05).sum() / x.size),
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


if __name__ == "__main__":
    setup_owkin_colors_palette()
