"""Plot file for the DP experiment."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fedeca.utils.constants import EXPERIMENTS_PATHS
from fedeca.utils.experiment_utils import load_dataframe_from_pickles

# TODO use Owkin's palette
# from fedeca.viz.plot import owkin_palette


def relative_error(x, y, absolute_error=False):
    """Compute the relative error."""
    if absolute_error:
        return np.abs(y - x) / np.abs(x)
    return np.linalg.norm(y - x) / np.linalg.norm(x)


names = {
    "Hazard Ratio": "hazard ratio",
    "Partial Log likelihood": "likelihood",
    "p-values": "p-values",
    "Propensity scores": "scores",
}
cmp = sns.color_palette("colorblind")
results = load_dataframe_from_pickles(
    EXPERIMENTS_PATHS["dp_results"] + "results_Pooled_equivalent_DP.pkl"
)

results_fl = results.loc[results["method"] == "FederatedIPTW", :]
results_pooled = results.loc[results["method"] == "IPTW", :]

errors = pd.DataFrame(
    data=np.abs(
        np.array(results_fl["exp(coef)"]) - np.array(results_pooled["exp(coef)"])
    )
    / np.abs(np.array(results_pooled["exp(coef)"])),
    columns=["hazard ratio"],
)

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
errors["epsilon"] = results_fl["dp_target_epsilon"].to_numpy()
errors["delta"] = results_fl["dp_target_delta"].to_numpy()


# fig, axarr = plt.subplots(1, 1, figsize=(10, 5))
# sns.boxplot(
#     data=errors, palette=sns.color_palette(owkin_palette.values(), 9), width=0.5
# )
# ax = sns.swarmplot(data=errors, color=".25", size=4)

# axarr.hlines(y=1e-2, xmin=-0.5, xmax=3.5, linewidth=2, color="r", linestyle="--")
# axarr.set_yscale("log")
# axarr.set_xticks(np.arange(errors.shape[1]), names)
# axarr.set_title("Pooled IPTW versus FedECA")
# axarr.set_ylabel("Relative error")
# axarr.set_ylim((1e-9, 1))
# plt.tight_layout()
# plt.savefig("pooled_equivalent.pdf")


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
deltas = [d for d in errors["delta"].unique() if not (np.isnan(d))]

for rel_error_name, col_name in names.items():
    fig, ax = plt.subplots()
    for i, d in enumerate(deltas):
        cdf = errors.loc[errors["delta"] == d]
        sns.lineplot(
            data=cdf,
            x="epsilon",
            y=col_name,
            label=rf"$\delta={d}$",
            linestyle=linestyles[::-1][i][1],
            ax=ax,
        )
    ax.set_xscale("log")
    if col_name in {"p-values", "likelihood"}:
        ax.set_yscale("log")
    xtick_values = np.logspace(-1.0, 1.6989700043360185, 5, base=10)
    xlabels = [str(round(v, 2)) for v in xtick_values]
    ax.set_xticks(xtick_values, xlabels)
    ax.axhline(
        1e-2,
        color="red",
        label=None,
        linestyle="--",
    )
    ax.set_xlim(0.1, 50.0)
    plt.legend()
    plt.xlim(0.1, 50.0)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Relative Errors")
    plt.tight_layout()
    plt.savefig(
        f"DP_relative_error_pooled_{rel_error_name}.pdf", dpi=100, bbox_inches="tight"
    )
    plt.clf()
