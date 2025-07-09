"""Script for creating SMD and power analysis figure."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats

from fedeca.utils.constants import EXPERIMENTS_PATHS
from fedeca.utils.experiment_utils import load_dataframe_from_pickles
from fedeca.viz.plot import FacetGridWithFigure, owkin_palette
from fedeca.viz.utils import adjust_legend_subtitles


def _parse_smd_results():
    filename = EXPERIMENTS_PATHS["smd_results"] / "results_smd_cov_shift.pkl"
    df_smd = load_dataframe_from_pickles(filename)

    df_smd = (
        df_smd.pipe(lambda df: df[df["method"].isin(["FedECA", "MAIC"])])
        .assign(cov_shift=lambda df: 0.5 * (df["overlap"] + 1))
        .filter(regex=r"cov_shift|method|rep_id|smd_\w+_X_")
        .assign(index=lambda df: range(df.shape[0]))
        .pipe(
            pd.wide_to_long,
            stubnames=["smd_weighted", "smd_raw"],
            i="index",
            j="covariate",
            sep="_",
            suffix=r"\w+",
        )
        .droplevel("index")
        .reset_index()
        .assign(index=lambda df: range(df.shape[0]))
        .pipe(
            pd.wide_to_long,
            stubnames="smd",
            i="index",
            j="weighted",
            sep="_",
            suffix=r"\w+",
        )
        .droplevel("index")
        .reset_index()
        .pipe(lambda df: df[df["method"].eq("FedECA") | df["weighted"].eq("weighted")])
        .assign(
            method=lambda df: df["method"].mask(df["weighted"].eq("raw"), "Unweighted"),
            smd=lambda df: df["smd"].abs(),
        )
    )
    return df_smd


def _parse_power_results():
    fname_power = "results_Power_and_type_one_error_analyses.pkl"
    df_power_cov_shift = load_dataframe_from_pickles(
        EXPERIMENTS_PATHS["power"] / "cov_shift" / fname_power
    )
    df_power_n_samples = load_dataframe_from_pickles(
        EXPERIMENTS_PATHS["power"] / "n_samples/" / fname_power
    )

    df_power = pd.concat(
        [
            df_power_n_samples.drop(columns="overlap"),
            df_power_cov_shift.drop(columns="n_samples"),
        ]
    )
    df_power["method"] = df_power["method"].replace(r"_.*", "", regex=True)
    method_recoding = {
        "IPTW": "FedECA*",
    }
    df_power["method"] = df_power["method"].replace(method_recoding)
    df_power["cov_shift"] = 0.5 * (df_power["overlap"] + 1)
    # Create dataframe for seaborn.FacetGrid
    df_power["y"] = df_power["cate"].replace({0.4: "power", 1.0: "type_one"})
    df_power["y_value"] = df_power["p"].lt(0.05).astype(int)
    df_power = df_power.melt(
        id_vars=["method", "variance_method", "y", "y_value"],
        value_vars=["n_samples", "cov_shift"],
        value_name="x_value",
        var_name="x",
    ).dropna(how="any")
    # Create column to be used as legend labels
    df_power["label"] = (
        df_power["method"].map(str) + " (" + df_power["variance_method"] + ")"
    )
    # Set category dtype, otherwise FacetGrid may bug with legend.
    # See https://github.com/mwaskom/seaborn/issues/2916
    df_power["label"] = df_power["label"].astype("category")
    power_labels = ["FedECA* (bootstrap)", "FedECA* (robust)", "MAIC (robust)"]
    df_power = df_power.pipe(
        lambda df: df[df["y"].ne("power") | df["label"].isin(power_labels)]
    )

    return df_power


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", action="store_true")
    args = parser.parse_args()

    # SMD results
    df_smd = _parse_smd_results()
    df_smd_box = df_smd.pipe(
        lambda df: df[
            df["cov_shift"].isin({0, 2})
            & df["covariate"].isin({f"X_{i}" for i in range(5)})
        ]
    )

    # Power analysis results
    df_power = _parse_power_results()

    # Colors and markers
    smd_colors = {
        "FedECA": owkin_palette["owkin_bright_blue"],
        "MAIC": owkin_palette["owkin_magenta"],
        "Unweighted": owkin_palette["owkin_mustard"],
    }
    smd_markers = {"FedECA": "s", "MAIC": "P", "Unweighted": "d"}

    power_colors = {
        "FedECA* (bootstrap)": owkin_palette["owkin_blue"],
        "FedECA* (robust)": owkin_palette["owkin_bright_blue"],
        "FedECA* (naive)": owkin_palette["owkin_teal"],
        "MAIC (bootstrap)": owkin_palette["owkin_pink"],
        "MAIC (robust)": owkin_palette["owkin_magenta"],
        "Unweighted (naive)": owkin_palette["owkin_mustard"],
    }

    # Figure size specs
    H_SMD = 4
    W_SMD_BOX = 4
    W_SMD_LINE = 5
    HSPACE = 0.15
    VSPACE = 0.15
    H_POWER = 5
    W_POWER = 7.5

    kw_smd_box = {
        "row": "cov_shift",
        "sharex": True,
        "margin_titles": True,
        "height": H_SMD / 2,
        "aspect": 2 * W_SMD_BOX / H_SMD,
        "legend_out": False,
    }

    kw_power = {
        "col": "x",
        "row": "y",
        "row_order": ["type_one", "power"],
        "col_order": ["cov_shift", "n_samples"],
        "height": H_POWER / 2,
        "aspect": W_POWER / H_POWER,
        "sharex": "col",
        "sharey": "row",
        "margin_titles": True,
        "legend_out": False,
    }

    fig = None
    if args.ensemble:
        # Figure layout:
        #
        # | SMD_LINE | SMD_BOX |
        # ----------------------
        # |    POWER    |      |
        fig_width = W_SMD_BOX + W_SMD_LINE + VSPACE
        fig_height = H_SMD + H_POWER + HSPACE
        fig = plt.figure(figsize=(fig_width, fig_height))
        # create subfigs
        subfigs = fig.subfigures(2, 1, hspace=HSPACE, height_ratios=[H_SMD, H_POWER])
        subfig_smd = subfigs[0].subfigures(
            1, 2, wspace=VSPACE, width_ratios=[W_SMD_LINE, W_SMD_BOX]
        )
        subfig_power = subfigs[1].subfigures(
            1, 2, wspace=VSPACE, width_ratios=[W_POWER, fig_width - W_POWER]
        )

        fig3a = subfig_smd[0]
        ax3a = fig3a.subplots()
        g3b = FacetGridWithFigure(df_smd_box, fig=subfig_smd[1], **kw_smd_box)
        g3c = FacetGridWithFigure(df_power, fig=subfig_power[0], **kw_power)
    else:
        # Separate figures
        fig3a, ax3a = plt.subplots(figsize=(W_SMD_LINE, H_SMD))
        g3b = sns.FacetGrid(df_smd_box, **kw_smd_box)
        g3c = sns.FacetGrid(df_power, **kw_power)

    # Plot SMD line plot
    sns.lineplot(
        df_smd,
        x="cov_shift",
        y="smd",
        hue="method",
        hue_order=smd_colors,
        palette=smd_colors,
        style="method",
        dashes=False,
        markers=smd_markers,
        markersize=8,
        ax=ax3a,
    )
    ax3a.legend(ncol=3)
    ax3a.axhline(10, color="black", linestyle="--", alpha=0.5)
    ax3a.set_xlabel("Covariate shift")
    ax3a.set_ylabel("Mean absolute SMD")
    ax3a.grid()

    # Plot SMD box plot
    g3b.map_dataframe(
        sns.boxplot,
        x="smd",
        y="covariate",
        hue="method",
        hue_order=smd_colors,
        palette=smd_colors,
        width=0.6,
        gap=0.3,
        whis=5,
        linewidth=0.7,
    )
    g3b.set_xlabels("Absolute SMD")
    g3b.set_ylabels("Covariate")
    g3b.set_titles(row_template="Covariate shift = {row_name}")
    for ax in g3b.axes.flat:
        ax.axvline(10, color="black", linestyle="--", alpha=0.2)
    g3b.add_legend(loc="upper right", ncol=1, bbox_to_anchor=(0.95, 0.95))

    # Plot power analysis
    g3c.map_dataframe(
        sns.lineplot,
        x="x_value",
        y="y_value",
        errorbar=("se", stats.norm.ppf(1 - 0.05 / 2)),
        hue="label",
        hue_order=power_colors,
        palette=power_colors,
        style="label",
        dashes=None,
        markers=True,
        markersize=8,
        err_style="bars",
    )
    g3c.set_titles(col_template="", row_template="")
    n_col = g3c.axes.shape[1]
    for i, ax in enumerate(g3c.axes.flat):
        if i // n_col == 0:
            ax.axhline(0.05, color="black", linestyle="dashed", alpha=0.2)
            ax.set_ylabel("Type I error")
            ax.set(yscale="log")
        if i // n_col == 1:
            ax.set_ylabel("Statistical power")
            ax.set_ylim(0, 1)
        if i % n_col == 0:
            ax.set_xlabel("Covariate shift")
        if i % n_col == 1:
            ax.set_xticks(range(300, 1200, 200))
            ax.set_xlabel("Number of samples")

    handles = list(g3c._legend_data.values())  # type: ignore
    labels = list(g3c._legend_data.keys())  # type: ignore
    handles.insert(0, Patch(visible=False))
    handles.insert(4, Patch(visible=False))
    labels.insert(0, "Requires federated learning")
    labels.insert(4, "Federated analytics")
    g3c._legend_data = dict(zip(labels, handles))  # type: ignore
    g3c.add_legend()
    sns.move_legend(g3c, "center left", bbox_to_anchor=(1, 0.5))
    adjust_legend_subtitles(g3c.legend)

    x, y = (0, 1) if args.ensemble else (0.05, 0.93)
    # Add subfigure labels (a) (b) (c)
    for figure, label in zip((fig3a, g3b.figure, g3c.figure), "abc"):
        # use spaces in title to add fixed padding.
        figure.suptitle(f"({label})" + " " * 5, fontfamily="serif", x=x, y=y)

    if fig is not None:
        fig.subplots_adjust(hspace=0.0)
        fig.tight_layout()
        fig.savefig("smd_power_ensemble.pdf", bbox_inches="tight")
    else:
        fig3a.tight_layout()
        fig3a.savefig("smd_curves.pdf", bbox_inches="tight")
        g3b.figure.tight_layout()
        g3b.figure.savefig("smd_cov_shift.pdf", bbox_inches="tight")
        g3c.figure.tight_layout()
        g3c.figure.savefig("power_curves.pdf", bbox_inches="tight")
