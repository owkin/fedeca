"""Script for creating power analysis figure."""
# %%
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats

from fedeca.utils.experiment_utils import load_dataframe_from_pickles
from fedeca.viz.plot import owkin_palette
from fedeca.viz.utils import adjust_legend_subtitles

# %%
# Raw results
BASE_PATH = "../results/power_analysis/"
FILE_OUTPUT = "results_Power_and_type_one_error_analyses.pkl"
df_res_cov_shift = load_dataframe_from_pickles(
    BASE_PATH + "cov_shift/" + FILE_OUTPUT
)
df_res_n_samples = load_dataframe_from_pickles(
    BASE_PATH + "n_samples/" + FILE_OUTPUT
)

# %%
# Aggregated results
df_plot = pd.concat(
    [
        df_res_n_samples.drop(columns="overlap"),
        df_res_cov_shift.drop(columns="n_samples"),
    ]
)
df_plot["method"] = df_plot["method"].replace(r"_.*", "", regex=True)
method_recoding = {
    "IPTW": "FedECA*",
}
df_plot["method"] = df_plot["method"].replace(method_recoding)
df_plot["cov_shift"] = 0.5 * (df_plot["overlap"] + 1)
# Create dataframe for seaborn.FacetGrid
df_plot["y"] = df_plot["cate"].replace({0.4: "power", 1.0: "type_one"})
df_plot["y_value"] = df_plot["p"].lt(0.05).astype(int)
df_plot = df_plot.melt(
    id_vars=["method", "variance_method", "y", "y_value"],
    value_vars=["n_samples", "cov_shift"],
    value_name="x_value",
    var_name="x",
).dropna(how="any")
# Create column to be used as legend labels
df_plot["label"] = df_plot["method"].map(str) + " (" + df_plot["variance_method"] + ")"
# Set category dtype, otherwise FacetGrid may bug with legend.
# See https://github.com/mwaskom/seaborn/issues/2916
df_plot["label"] = df_plot["label"].astype("category")

# %%
power_labels = [
    "FedECA* (bootstrap)",
    "FedECA* (robust)",
    "MAIC (robust)",
]
g = sns.FacetGrid(
    df_plot.query("y != 'power' | label in @power_labels"),
    col="x",
    row="y",
    row_order=["type_one", "power"],
    col_order=["cov_shift", "n_samples"],
    height=3,
    aspect=1.5,  # type: ignore
    sharex="col",  # type: ignore
    sharey="row",  # type: ignore
    margin_titles=True,
)
g.map_dataframe(
    sns.lineplot,
    x="x_value",
    y="y_value",
    errorbar=("se", stats.norm.ppf(1 - 0.05 / 2)),
    hue="label",
    hue_order=[
        "FedECA* (bootstrap)",
        "FedECA* (robust)",
        "FedECA* (naive)",
        "MAIC (bootstrap)",
        "MAIC (robust)",
        "Unweighted (naive)",
    ],
    style="label",
    dashes=None,
    markers=True,
    markersize=8,
    err_style="bars",
    palette=[list(owkin_palette.values())[i] for i in (3, 5, 1, 0, 4, 2)],
)
g.set_titles(col_template="", row_template="")
n_col = g.axes.shape[1]
for i, ax in enumerate(g.axes.flat):
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

handles = list(g._legend_data.values())
labels = list(g._legend_data.keys())
handles.insert(0, Patch(visible=False))
handles.insert(4, Patch(visible=False))
labels.insert(0, "Requires federated learning")
labels.insert(4, "Federated analytics")
g._legend_data = dict(zip(labels, handles))
g.add_legend()
sns.move_legend(g, "center right", bbox_to_anchor=(0.94, 0.5))
adjust_legend_subtitles(g.legend)
g.figure.suptitle("(c)", fontfamily="serif", x=0.05)
g.figure.set_dpi(300)

# %%
g.savefig("fedeca_power_and_type_one_error.pdf", bbox_inches="tight", dpi=300)
