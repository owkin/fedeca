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
    / "results_Robust_Pooled_Equivalent_nb_clients.pkl"
)

n_clients = [2, 3, 5, 10]

errors = {}
for name in names:
    errors[name] = pd.DataFrame()
for n_client in n_clients:
    results_tmp = results.loc[results["n_clients"] == n_client, :]
    results_fl = results_tmp.loc[results_tmp["method"] == "FedECA", :]
    results_pooled = results_tmp.loc[results_tmp["method"] == "IPTW", :]

    errors["Hazard Ratio"][n_client] = pd.DataFrame(
        data=np.abs(
            np.array(results_fl["exp(coef)"]) - np.array(results_pooled["exp(coef)"])
        )
        / np.abs(np.array(results_pooled["exp(coef)"])),
        columns=["hazard ratio"],
    )

    errors["Partial Log likelihood"][n_client] = np.abs(
        np.array(results_fl["log_likelihood"])
        - np.array(results_pooled["log_likelihood"])
    ) / np.abs(np.array(results_pooled["log_likelihood"]))

    errors["p-values"][n_client] = np.abs(
        np.array(results_fl["p"]) - np.array(results_pooled["p"])
    ) / np.abs(np.array(results_pooled["p"]))

    errors["Propensity scores"][n_client] = np.array(
        [
            relative_error(
                np.array(results_pooled["propensity_scores"].iloc[i]),
                np.array(results_fl["propensity_scores"].iloc[i]),
            )
            for i in range(results_fl.shape[0])
        ]
    )

dict_ylim = {
    "Hazard Ratio": (1e-8, 1),
    "Partial Log likelihood": (1e-10, 1),
    "p-values": (1e-6, 1),
    "Propensity scores": (1e-6, 1),
}


fig, axarr = plt.subplots(2, 2, figsize=(15, 7.5))
j = 0
for i, name in enumerate(names):
    print(i)
    if i > 1:
        j = 1
    sns.boxplot(
        data=errors[name],
        palette=sns.color_palette(owkin_palette.values(), 9),
        width=0.5,
        ax=axarr[i % 2, j],
    )
    sns.swarmplot(data=errors[name], color=".25", size=2, ax=axarr[i % 2, j])

    axarr[i % 2, j].hlines(
        y=1e-2, xmin=-0.5, xmax=3.5, linewidth=2, color="r", linestyle="--"
    )
    axarr[i % 2, j].set_yscale("log")
    axarr[i % 2, j].set_xticks(np.arange(errors[name].shape[1]), n_clients)
    axarr[i % 2, j].set_title(f"{name}")
    axarr[i % 2, j].set_ylabel("Relative error")
    axarr[i % 2, j].set_ylim(dict_ylim[name])
plt.tight_layout()
plt.savefig("robust_pooled_equivalent_nb_clients.png")
