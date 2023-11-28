import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fedeca.competitors import MatchingAjudsted, NaiveComparison, PooledIPTW
from fedeca.utils.experiment_utils import param_grid_from_dict, single_experiment
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
    iptw = PooledIPTW(
        treated_col=TREATED,
        event_col=EVENT,
        duration_col=TIME,
        effect="ATE",
    )
    maic = MatchingAjudsted(
        treated_col=TREATED,
        event_col=EVENT,
        duration_col=TIME,
    )
    naive = NaiveComparison(
        treated_col=TREATED,
        event_col=EVENT,
        duration_col=TIME,
    )
    models = {
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
                models=models,
                treated_col=TREATED,
                event_col=EVENT,
                duration_col=TIME,
            )
            for _ in range(getattr(row, "n_repeats"))
        ]
    results = pd.concat(results)

    with open(args.output, "wb") as f1:
        pickle.dump(results, f1)
