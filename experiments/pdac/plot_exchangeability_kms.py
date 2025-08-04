"""Script for creating exchangeability KM curves."""

import pickle

import numpy as np
import seaborn as sns

from fedeca.utils.constants import EXPERIMENTS_PATHS
from fedeca.viz.plot import plot_survival_function

if __name__ == "__main__":
    with open(EXPERIMENTS_PATHS["pdac"] / "rw_pdac_exchangeability_km.pkl", "rb") as f:
        df_res = pickle.load(f)

    g = (
        sns.FacetGrid(
            df_res,
            col="treatment",
            row="pair",
            hue="center",
            margin_titles=True,
            sharex=False,
            height=4,
            aspect=1.3,
        )
        .map_dataframe(plot_survival_function)
        .set_axis_labels("Survival time (month)", "Probability of survival")
    )
    for ax in g.axes.flat:
        ax.legend()
        _, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(0, end, 10))

    g.figure.savefig("fed_km_exchangeability.pdf")
