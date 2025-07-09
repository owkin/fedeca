"""Script for creating synthetic data SMD results."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from fedeca.utils.constants import EXPERIMENTS_PATHS
from fedeca.utils.experiment_utils import load_dataframe_from_pickles
from fedeca.viz.plot import owkin_palette

# Load raw results
sns.set_theme(rc={"figure.figsize": (11.69, 8.27)})
plt.style.use("default")

# plot lines for smd as a function of covariate shift
dict_colors = {
    "FedECA": list(owkin_palette.values())[5],
    "MAIC": list(owkin_palette.values())[4],
    "Unweighted": list(owkin_palette.values())[2],
}
dict_markers = {"FedECA": "s", "MAIC": "P", "Unweighted": "d"}


fname = EXPERIMENTS_PATHS["smd_results"] / "results_smd_cov_shift.pkl"
df_res = load_dataframe_from_pickles(fname)

df_res["cov_shift"] = 0.5 * (df_res["overlap"] + 1)
df_res = df_res.loc[df_res["method"] != "IPTW"]
df_res = df_res.loc[df_res["method"] != "Unweighted"]

# Preprocess results
df_res = df_res.filter(regex=r"cov_shift|method|rep_id|smd_\w+_X_")

df_res = (
    pd.wide_to_long(
        df_res.reset_index(drop=True).reset_index(),
        stubnames=["smd_weighted", "smd_raw"],
        i="index",
        j="covariate",
        sep="_",
        suffix=r"\w+",
    )
    .reset_index()
    .drop(columns="index")
)
df_res = (
    pd.wide_to_long(
        df_res.reset_index(),
        stubnames="smd",
        i="index",
        j="weighted",
        sep="_",
        suffix=r"\w+",
    )
    .reset_index()
    .drop(columns="index")
)

df_res = df_res.loc[
    np.logical_or(df_res["method"] == "FedECA", df_res["weighted"] == "weighted")
]
df_res.loc[df_res["weighted"] == "raw", "method"] = "Unweighted"
df_res["smd"] = np.abs(df_res["smd"])


# Plot
g = sns.FacetGrid(
    df_res[
        df_res["cov_shift"].isin([0, 2])
        & df_res["covariate"].isin(["X_0", "X_1", "X_2", "X_3", "X_4"])
    ],
    row=None,
    col="cov_shift",
    margin_titles=True,
)
g.map_dataframe(
    sns.boxplot,
    x="smd",
    y="covariate",
    hue="method",
    width=0.6,
    palette=list(owkin_palette.values())[4:5]
    + list(owkin_palette.values())[5:6]
    + list(owkin_palette.values())[2:3],
    gap=0.3,
    whis=5,
    linewidth=0.7,
)
g.set_xlabels("Absolute Standardized \n mean difference")
g.set_ylabels("Covariate")
g.set_titles(row_template="{row_name}", col_template="Covariate shift = {col_name}")
for ax in g.axes.flat:
    ax.axvline(10, color="black", linestyle="--", alpha=0.2)
# g.add_legend(title="Method")
g.figure.suptitle("(b)", fontfamily="serif", x=0.05)
g.savefig("smd_cov_shift.pdf", bbox_inches="tight")


df_smd = df_overlap = (
    df_res.groupby(["method", "cov_shift", "weighted"])
    .agg(
        smd=pd.NamedAgg(column="smd", aggfunc=lambda x: np.abs(x).sum() / x.size),
        lower=pd.NamedAgg(
            column="smd",
            aggfunc=lambda x: np.abs(x).sum() / x.size
            - stats.norm.ppf(0.975) * np.sqrt(np.var(np.abs(x)) / x.size),
        ),
        upper=pd.NamedAgg(
            column="smd",
            aggfunc=lambda x: np.abs(x).sum() / x.size
            + stats.norm.ppf(0.975) * np.sqrt(np.var(np.abs(x)) / x.size),
        ),
    )
    .reset_index()
)

plt.style.use("default")
matplotlib.rc("xtick", labelsize=18)
matplotlib.rc("ytick", labelsize=18)
plt.rcParams["legend.title_fontsize"] = "20"


fig, axarr = plt.subplots(1, 1, figsize=(11.69, 8.27))
selected_methods = ["FedECA", "MAIC", "Unweighted"]
for _, method in enumerate(selected_methods):
    axarr.plot(
        df_smd[df_smd["method"] == method]["cov_shift"],
        df_smd[df_smd["method"] == method]["smd"],
        marker=dict_markers[method],
        label=method,
        color=dict_colors[method],
        markersize=10,
    )
    axarr.fill_between(
        df_smd[df_smd["method"] == method]["cov_shift"],
        df_smd[df_smd["method"] == method]["lower"],
        df_smd[df_smd["method"] == method]["upper"],
        color=dict_colors[method],
        alpha=0.4,
    )
plt.axhline(y=10, color="black", linestyle="--", alpha=0.2, linewidth=5)
axarr.set_xlabel("Covariate shift", fontsize=20)
axarr.set_ylabel("Mean absolute standardized \n mean difference", fontsize=20)

axarr.grid()
fig.legend(bbox_to_anchor=(0.81, 1.03), ncol=3, title="Method", fontsize=20)
fig.suptitle("(a)", fontfamily="serif", x=0.05, fontsize=20)
fig.savefig("smd_curves.pdf", bbox_inches="tight")
