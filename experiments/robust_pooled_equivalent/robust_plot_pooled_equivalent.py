"""Plot file for the pooled equivalent experiment."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fedeca.utils.constants import EXPERIMENTS_PATHS
from fedeca.utils.experiment_utils import load_dataframe_from_pickles
from fedeca.viz.plot import owkin_palette


def relative_error(x, y, absolute_error=False):
    """Compute the relative error."""
    if absolute_error:
        return np.abs(y - x) / np.abs(x)
    else:
        return np.linalg.norm(y - x) / np.linalg.norm(x)


names = ["Hazard Ratio", "Partial Log likelihood", "p-values", "Propensity scores"]
cmp = sns.color_palette("colorblind")
results = load_dataframe_from_pickles(
    EXPERIMENTS_PATHS["robust_pooled_equivalence"]
    + "results_Robust_Pooled_Equivalent.pkl"
)

results_fl = results.loc[results["method"] == "FedECA", :]
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


fig, axarr = plt.subplots(1, 1, figsize=(10, 5))
sns.boxplot(
    data=errors, palette=sns.color_palette(owkin_palette.values(), 9), width=0.5
)
ax = sns.swarmplot(data=errors, color=".25", size=4)

axarr.hlines(y=1e-2, xmin=-0.5, xmax=3.5, linewidth=2, color="r", linestyle="--")
axarr.set_yscale("log")
axarr.set_xticks(np.arange(errors.shape[1]), names)
axarr.set_title("Pooled IPTW versus FedECA")
axarr.set_ylabel("Relative error")
axarr.set_ylim((1e-9, 1))
plt.tight_layout()
plt.savefig("robust_pooled_equivalent.pdf")
