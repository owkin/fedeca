import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiments.synthetic import single_experiment
from fedeca.competitors import MatchingAjudsted, NaiveComparison, PooledIPTW
from fedeca.utils.experiment_utils import param_grid_from_dict
from fedeca.utils.survival_utils import CoxData

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="Name of the config file",
        action="store",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output file",
        action="store",
    )
    args = parser.parse_args()
    if args.config is None:
        args.config = "hr_estimation_power.yaml"
    if args.output is None:
        output = re.sub(r".*/", "", args.config)
        args.output = "results_" + re.sub(r"\.yaml$", ".pkl", output)

    with open(
        Path(__file__).parent.parent / "config" / args.config,
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    TREATED = "treatment"
    EVENT = "event"
    TIME = "time"
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

    naive = config_base.copy()
    naive["_target_"] = NaiveComparison

    model_configs = {
        "IPTW": iptw,
        "MAIC": maic,
        "Naive": naive,
    }

    config_experiment = config["parameters"]["experiments"]
    df_params = param_grid_from_dict(config_experiment)
    seeds = np.random.SeedSequence(config["seed"]).generate_state(df_params.shape[0])

    results = []
    for i, row in enumerate(df_params.itertuples()):
        coxdata = CoxData(
            n_samples=1,
            cate=getattr(row, "cate"),
            propensity="linear",
            seed=seeds[i],
            percent_ties=None,
            ndim=getattr(row, "n_covariates"),
            standardize_features=False,
        )
        results = results + [
            single_experiment(
                coxdata,
                n_samples=getattr(row, "n_samples"),
                model_configs=model_configs,
                treated_col=TREATED,
                event_col=EVENT,
                duration_col=TIME,
            )
            for _ in range(getattr(row, "n_repeats"))
        ]
    results = pd.concat(results)

    with open(args.output, "wb") as f1:
        pickle.dump(results, f1)
