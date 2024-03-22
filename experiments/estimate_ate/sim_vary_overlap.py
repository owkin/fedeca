"""Simulation data for FedECA."""
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiments.synthetic import single_experiment
from fedeca.competitors import (
    CovariateAdjusted,
    MatchingAjudsted,
    NaiveComparison,
    PooledIPTW,
)
from fedeca.utils.survival_utils import CoxData

if __name__ == "__main__":
    with open(
        Path(__file__).parent.parent / "config/pooled_vary_overlap.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    TREATED = "treatment"
    EVENT = "event"
    TIME = "time"
    PS = "propensity_scores"
    config_base: dict[str, Any] = {
        "treated_col": TREATED,
        "event_col": EVENT,
        "duration_col": TIME,
    }

    iptw = config_base.copy()
    iptw["_target_"] = PooledIPTW
    iptw["effect"] = "ATE"

    maic = config_base.copy()
    maic["_target_"] = MatchingAjudsted

    covadjust = config_base.copy()
    covadjust["_target_"] = CovariateAdjusted

    naive = config_base.copy()
    naive["_target_"] = NaiveComparison

    model_configs = {
        "IPTW": iptw,
        "MAIC": maic,
        # "CovAdj": covadjust,
        "Naive": naive,
        "OracleIPTW": iptw,
    }

    config_experiment = config["parameters"]["experiments"]
    results = []
    list_overlap = config_experiment["overlap"]
    seeds = np.random.SeedSequence(config["seed"]).generate_state(len(list_overlap))
    for i, overlap in enumerate(list_overlap):
        coxdata = CoxData(
            n_samples=config_experiment["n_samples"],
            cate=0.0,
            propensity="linear",
            seed=seeds[i],
            percent_ties=None,
            ndim=config_experiment["n_covariates"],
            standardize_features=False,
            overlap=overlap,
            prop_treated=config_experiment["prop_treated"],
        )
        for j in range(config_experiment["n_repeats"]):
            res_single_exp = single_experiment(
                coxdata,
                n_samples=config_experiment["n_samples"],
                model_configs=model_configs,
                treated_col=TREATED,
                event_col=EVENT,
                duration_col=TIME,
                ps_col=PS,
            )

            results.append(res_single_exp.assign(overlap=overlap, exp_id=j))

    results = pd.concat(results)

    with open("results_sim_vary_overlap.pkl", "wb") as f1:
        pickle.dump(results, f1)
