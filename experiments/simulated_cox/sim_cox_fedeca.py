"""Simulation data for FedECA."""
import pickle

import numpy as np
import pandas as pd
import torch
import yaml

from fedeca import FedECA, PooledIPTW
from fedeca.utils.constants import EXPERIMENTS_PATHS
from fedeca.utils.survival_utils import CoxData


def relative_error(x, y, absolute_error=False):
    """Compute the relative error."""
    if absolute_error:
        return np.abs(y - x) / np.abs(x)
    else:
        return np.linalg.norm(y - x) / np.linalg.norm(x)


def simulated_fl_benchmark(
    nb_client=2,
    n_samples=500,
    percent_ties=None,
    n_repeat=5,
    n_covariates=10,
    group_treated=False,
    nb_rounds_list=[10, 10],
):
    """Execute the experiment."""
    error = {"weights": [], "treatment effect": [], "p-values": [], "likelihood": []}
    seeds = []
    for k in range(n_repeat):
        seed = 123 + k
        print("Lauching data generation with seed", seed)
        # Simulate data
        coxdata = CoxData(
            n_samples=n_samples,
            cate=1.0,
            seed=seed,
            percent_ties=percent_ties,
            ndim=n_covariates,
            propensity="linear",
        )
        X, times, censoring, treat_alloc = coxdata.generate_data()
        col_X = ["X_%i" % i for i in range(X.shape[1])]

        data = np.concatenate(
            [
                X,
                times[:, np.newaxis],
                censoring[:, np.newaxis],
                treat_alloc[:, np.newaxis],
            ],
            axis=1,
        )

        data = pd.DataFrame(
            data=data, columns=col_X + ["time", "event", "treatment_allocation"]
        )
        # define treatment allocation
        treatment_allocation = "treatment_allocation"

        print("Computing propensity weights on pooled data")
        # Instantiate IPTW class
        # We can specify the type of effect we want to estimate

        iptw = PooledIPTW(
            treated_col=treatment_allocation,
            event_col="event",
            duration_col="time",
            effect="ATE",
        )

        # We can now estimate the treatment effect
        iptw.fit(data)
        df = data.drop(columns=["weights"])
        df["treatment_allocation"] = df["treatment_allocation"].values.astype("uint8")
        print("Computing propensity weights on distributed data")

        fl_iptw = FedECA(
            ndim=n_covariates,
            treated_col="treatment_allocation",
            duration_col="time",
            event_col="event",
            num_rounds_list=nb_rounds_list,
            dtype="float64",
        )
        if group_treated:
            split_method = ("split_control_over_centers",)
            split_method_kwargs = {"treatment_info": "treated"}
        else:
            split_method = "uniform"
            split_method_kwargs = None
        fl_iptw.fit(df, None, nb_client, split_method, split_method_kwargs)

        m = fl_iptw.propensity_model

        Xprop = torch.from_numpy(X)
        with torch.no_grad():
            propensity_scores = m(Xprop)

        propensity_scores = propensity_scores.detach().numpy().flatten()
        weights = df["treatment_allocation"] * 1.0 / propensity_scores + (
            1 - df["treatment_allocation"]
        ) * 1.0 / (1.0 - propensity_scores)

        # L2 error || fl_weights - pooled_weights ||_2
        error["weights"].append(relative_error(data["weights"], weights))
        error["p-values"].append(
            relative_error(
                iptw.results_["p"].iloc[0],
                fl_iptw.results_["p"].iloc[0],
                absolute_error=True,
            )
        )

        error["treatment effect"].append(
            relative_error(
                iptw.results_["coef"].iloc[0],
                fl_iptw.results_["coef"].iloc[0],
                absolute_error=True,
            )
        )
        error["likelihood"].append(
            relative_error(iptw.log_likelihood_, fl_iptw.ll, absolute_error=True)
        )

        seeds.append(seed)

    return (
        nb_client,
        percent_ties,
        n_covariates,
        n_repeat,
        np.array(seeds),
        error,
        group_treated,
        nb_rounds_list,
    )


if __name__ == "__main__":
    with open("../config/pooled_equivalent_hardcore.yaml", "r") as file:
        config_experiment = yaml.safe_load(file)
    results = []

    for nb_client in config_experiment["parameters"]["experiments"]["nb_clients"]:
        results.append(
            simulated_fl_benchmark(
                nb_client=nb_client,
                percent_ties=config_experiment["parameters"]["experiments"][
                    "percent_ties"
                ],
                n_repeat=config_experiment["parameters"]["experiments"]["n_repeat"],
                n_covariates=config_experiment["parameters"]["experiments"][
                    "n_covariates"
                ],
                n_samples=config_experiment["parameters"]["experiments"]["n_samples"],
                group_treated=config_experiment["parameters"]["experiments"][
                    "group_treated"
                ],
                nb_rounds_list=config_experiment["parameters"]["fedeca"][
                    "nb_rounds_list"
                ],
            )
        )
    results = pd.DataFrame(
        data=results,
        columns=[
            "nb_clients",
            "percent_ties",
            "n_covariates",
            "n_repeat",
            "seeds",
            "error",
            "group treated",
            "nb rounds",
        ],
    )
    with open(
        EXPERIMENTS_PATHS["pooled_equivalent"]
        + "results_sim_cox_pooled_equivalent_hardcore.pkl",
        "wb",
    ) as f1:
        pickle.dump(results, f1)
