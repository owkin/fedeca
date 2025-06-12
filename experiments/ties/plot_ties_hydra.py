"""Plot file for the DP experiment."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fedeca.utils.constants import EXPERIMENTS_PATHS
from fedeca.utils.experiment_utils import load_dataframe_from_pickles
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np



def relative_error(x, y, absolute_error=False):
    """Compute the relative error."""
    if absolute_error:
        return np.abs(y - x) / np.abs(x)
    else:
        return np.linalg.norm(y - x) / np.linalg.norm(x)


names = {
    "Hazard Ratio": "hazard ratio",
    "Partial Log likelihood": "likelihood",
    "p-values": "p-values",
    "Propensity scores": "scores",
}
cmp = sns.color_palette("colorblind")

results = load_dataframe_from_pickles(
    EXPERIMENTS_PATHS["robust_pooled_equivalence_ties"] + "results_Pooled_equivalent_ties.pkl"
)

results.loc[results["percent_ties"].isnull(), "percent_ties"] = 0.
results_fl = results.loc[results["method"] == "FederatedIPTW", :]
results_pooled = results.loc[results["method"] == "IPTW", :]

errors = pd.DataFrame(
    data=np.abs(
        np.array(results_fl["exp(coef)"]) - np.array(results_pooled["exp(coef)"])
    )
    / np.abs(np.array(results_pooled["exp(coef)"])),
    columns=["hazard ratio"],
)
errors["percent_ties"] = results_fl["percent_ties"].tolist()

errors["likelihood"] = np.abs(
    np.array(results_fl["log_likelihood"]) - np.array(results_pooled["log_likelihood"])
) / np.abs(np.array(results_pooled["log_likelihood"]))

errors["p-values"] = np.abs(
    np.array(results_fl["p"]) - np.array(results_pooled["p"])
) / np.abs(np.array(results_pooled["p"]))

errors["scores"] = np.array(
    [
        relative_error(
            np.array(results_pooled["propensity_scores"].iloc[i]),
            np.array(results_fl["propensity_scores"].iloc[i]),
        )
        for i in range(results_fl.shape[0])
    ]
)


linestyle_str = [
    ("solid", "solid"),  # Same as (0, ()) or '-'
    ("dotted", "dotted"),  # Same as (0, (1, 1)) or ':'
    ("dashed", "dashed"),  # Same as '--'
    ("dashdot", "dashdot"),
]
linestyle_tuple = [
    ("loosely dotted", (0, (1, 10))),
    ("densely dotted", (0, (1, 1))),
    ("loosely dashed", (0, (5, 10))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]
linestyles = linestyle_tuple + linestyle_str

for rel_error_name, col_name in names.items():
    fig, ax = plt.subplots()

    cdf = errors
    sns.lineplot(
        data=cdf,
        x="percent_ties",
        y=col_name,
        linestyle=linestyles[::-1][0][1],
        ax=ax,
    )
    ax.set_xticks(np.arange(0, 1.1, 0.1))  # Integer number of ticks (0 to 1 by 0.1)

    # Format ticks as percentages with one decimal
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    if col_name == "p-values" or col_name == "likelihood":
        ax.set_yscale("log")

    ax.axhline(
        1e-2,
        color="red",
        label=None,
        linestyle="--",
    )

    plt.legend()

    plt.xlabel("Global % of ties")
    plt.ylabel("Relative Errors")
    plt.tight_layout()

    plt.savefig(
        f"ties_{rel_error_name}.pdf", dpi=100, bbox_inches="tight"
    )
    plt.clf()
