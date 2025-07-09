"""File for experiment figure plots."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from seaborn._base import categorical_order
from seaborn._core.data import handle_data_source
from seaborn.axisgrid import FacetGrid
from seaborn.utils import _disable_autolayout

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


# Hacky solution, diffs against the original __init__ are marked with NOTE
# https://stackoverflow.com/a/76175112
class FacetGridWithFigure(FacetGrid):
    """Seaborn FacetGrid that accepts existing figure object."""

    def __init__(
        self,
        data,
        *,
        row=None,
        col=None,
        hue=None,
        col_wrap=None,
        sharex=True,
        sharey=True,
        height=3,
        aspect=1,
        palette=None,
        row_order=None,
        col_order=None,
        hue_order=None,
        hue_kws=None,
        dropna=False,
        legend_out=True,
        despine=True,
        margin_titles=False,
        xlim=None,
        ylim=None,
        subplot_kws=None,
        gridspec_kws=None,
        fig=None,
    ):
        # NOTE: skip FacetGrid init
        super(FacetGrid, self).__init__()
        data = handle_data_source(data)

        # Determine the hue facet layer information
        hue_var = hue
        hue_names = None if hue is None else categorical_order(data[hue], hue_order)

        colors = self._get_palette(data, hue, hue_order, palette)

        # Set up the lists of names for the row and column facet variables
        row_names = [] if row is None else categorical_order(data[row], row_order)

        col_names = [] if col is None else categorical_order(data[col], col_order)

        # Additional dict of kwarg -> list of values for mapping the hue var
        hue_kws = hue_kws if hue_kws is not None else {}

        # Make a boolean mask that is True anywhere there is an NA
        # value in one of the faceting variables, but only if dropna is True
        none_na = np.zeros(len(data), bool)
        if dropna:
            row_na = none_na if row is None else data[row].isna()
            col_na = none_na if col is None else data[col].isna()
            hue_na = none_na if hue is None else data[hue].isna()
            not_na = ~(row_na | col_na | hue_na)
        else:
            not_na = ~none_na

        # Compute the grid shape
        ncol = 1 if col is None else len(col_names)
        nrow = 1 if row is None else len(row_names)
        self._n_facets = ncol * nrow

        self._col_wrap = col_wrap
        if col_wrap is not None:
            if row is not None:
                err = "Cannot use `row` and `col_wrap` together."
                raise ValueError(err)
            ncol = col_wrap
            nrow = int(np.ceil(len(col_names) / col_wrap))
        self._ncol = ncol
        self._nrow = nrow

        # Calculate the base figure size
        # This can get stretched later by a legend
        # TODO this doesn't account for axis labels
        figsize = (ncol * height * aspect, nrow * height)

        # Validate some inputs
        if col_wrap is not None:
            margin_titles = False

        # Build the subplot keyword dictionary
        subplot_kws = {} if subplot_kws is None else subplot_kws.copy()
        gridspec_kws = {} if gridspec_kws is None else gridspec_kws.copy()
        if xlim is not None:
            subplot_kws["xlim"] = xlim
        if ylim is not None:
            subplot_kws["ylim"] = ylim

        # --- Initialize the subplot grid

        # NOTE: use existing figure if provided
        if fig is None:
            with _disable_autolayout():
                fig = plt.figure(figsize=figsize)

        if col_wrap is None:
            kwargs = {
                "squeeze": False,
                "sharex": sharex,
                "sharey": sharey,
                "subplot_kw": subplot_kws,
                "gridspec_kw": gridspec_kws,
            }

            axes = fig.subplots(nrow, ncol, **kwargs)

            if col is None and row is None:
                axes_dict = {}
            elif col is None:
                axes_dict = dict(zip(row_names, axes.flat))
            elif row is None:
                axes_dict = dict(zip(col_names, axes.flat))
            else:
                facet_product = product(row_names, col_names)
                axes_dict = dict(zip(facet_product, axes.flat))

        else:
            # If wrapping the col variable we need to make the grid ourselves
            if gridspec_kws:
                warnings.warn("`gridspec_kws` ignored when using `col_wrap`")

            n_axes = len(col_names)
            axes = np.empty(n_axes, object)
            axes[0] = fig.add_subplot(nrow, ncol, 1, **subplot_kws)
            if sharex:
                subplot_kws["sharex"] = axes[0]
            if sharey:
                subplot_kws["sharey"] = axes[0]
            for i in range(1, n_axes):
                axes[i] = fig.add_subplot(nrow, ncol, i + 1, **subplot_kws)

            axes_dict = dict(zip(col_names, axes))

        # --- Set up the class attributes

        # Attributes that are part of the public API but accessed through
        # a  property so that Sphinx adds them to the auto class doc
        self._figure = fig
        self._axes = axes
        self._axes_dict = axes_dict
        self._legend = None

        # Public attributes that aren't explicitly documented
        # (It's not obvious that having them be public was a good idea)
        self.data = data
        self.row_names = row_names
        self.col_names = col_names
        self.hue_names = hue_names
        self.hue_kws = hue_kws

        # Next the private variables
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col

        self._margin_titles = margin_titles
        self._margin_titles_texts = []
        self._col_wrap = col_wrap
        self._hue_var = hue_var
        self._colors = colors
        self._legend_out = legend_out
        self._legend_data = {}
        self._x_var = None
        self._y_var = None
        self._sharex = sharex
        self._sharey = sharey
        self._dropna = dropna
        self._not_na = not_na

        # --- Make the axes look good

        self.set_titles()
        self.tight_layout()

        if despine:
            self.despine()

        if sharex in [True, "col"]:
            for ax in self._not_bottom_axes:
                for label in ax.get_xticklabels():
                    label.set_visible(False)
                ax.xaxis.offsetText.set_visible(False)
                ax.xaxis.label.set_visible(False)

        if sharey in [True, "row"]:
            for ax in self._not_left_axes:
                for label in ax.get_yticklabels():
                    label.set_visible(False)
                ax.yaxis.offsetText.set_visible(False)
                ax.yaxis.label.set_visible(False)

    # NOTE: subfigures don't have a tight layout option
    def tight_layout(self):
        pass


if __name__ == "__main__":
    setup_owkin_colors_palette()
