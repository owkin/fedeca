"""Script for creating the histogram of weights."""

import pickle
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fedeca.utils.constants import EXPERIMENTS_PATHS

if __name__ == "__main__":
    # Load histogram cuts and weight counts
    with open(EXPERIMENTS_PATHS["pdac"] / "rw_pdac_histogram.pkl", "rb") as f:
        df_res = pickle.load(f)
        df_res = df_res.dropna(how="any", subset="Count")

    for center in ("FFCD", "IDIBGI"):
        # Create pseudo weights to reproduce the figure with `sns.histplot`
        df_weights = (
            df_res[df_res["center"].eq(center)]
            # Repeat each row of dataframe by value in the column `col_count`
            .pipe(lambda df: df.loc[df.index.repeat(df["Count"])])
            .rename(columns={"Cut": "weights"})
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        N = 25 + 1
        bins = np.linspace(1.0, 7.0, N)
        sns.histplot(data=df_weights, x="weights", hue="treatment", bins=bins, ax=ax)
        # background pattern
        for b0, b1, color in zip(bins[:-1], bins[1:], cycle(["crimson", "lightblue"])):
            ax.axvspan(b0, b1, color=color, alpha=0.1, zorder=0)
        ax.get_legend().set_title("")
        fig.savefig(
            f"histograms_weights_{center.lower()}.pdf", bbox_inches="tight", dpi=300
        )
