import pandas as pd
import seaborn as sns

from fedeca.utils.experiment_utils import load_dataframe_from_pickles
from fedeca.viz.plot import owkin_palette

# Load raw results
fname = (
    "/home/owkin/project/results_experiments/smd_cov_shift/results_smd_cov_shift.pkl"
)
df_res = load_dataframe_from_pickles(fname)

df_res["cov_shift"] = 0.5 * (df_res["overlap"] + 1)

# Preprocess results
df = df_res.filter(regex=r"cov_shift|method|rep_id|smd_\w+_X_")
df = (
    pd.wide_to_long(
        df.reset_index(drop=True).reset_index(),
        stubnames=["smd_weighted", "smd_raw"],
        i="index",
        j="covariate",
        sep="_",
        suffix=r"\w+",
    )
    .reset_index()
    .drop(columns="index")
)
df = (
    pd.wide_to_long(
        df.reset_index(),
        stubnames="smd",
        i="index",
        j="weighted",
        sep="_",
        suffix=r"\w+",
    )
    .reset_index()
    .drop(columns="index")
)
df["weighted"] = df["weighted"].replace({"weighted": True, "raw": False})
method_recoding = {
    "FedECA": "FedECA",
    "IPTW": "IPTW",
    "MAIC": "MAIC",
}
df["method"] = df["method"].replace(method_recoding)

# Plot
g = sns.FacetGrid(
    df[
        df["cov_shift"].isin([0, 2])
        & df["covariate"].isin(["X_0", "X_1", "X_2", "X_3", "X_4"])
    ],
    col="method",
    col_order=["IPTW", "FedECA", "MAIC"],
    row="cov_shift",
    height=3.5,  # type: ignore
    aspect=0.8,  # type: ignore
    margin_titles=True,
)
g.map_dataframe(
    sns.boxplot,
    x="smd",
    y="covariate",
    hue="weighted",
    width=0.3,
    palette=owkin_palette.values(),
)
g.set_xlabels("Standardized mean difference")
g.set_ylabels("Covariate")
g.set_titles(col_template="{col_name}", row_template="Covariate shift = {row_name}")
for ax in g.axes.flat:
    ax.axvline(0, color="black", linestyle="--", alpha=0.2)
g.add_legend(title="Weighted")
g.savefig("smd_cov_shift.pdf", bbox_inches="tight")
