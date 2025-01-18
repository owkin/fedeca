"""Runs the propensity model training part with DP."""
import sys
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from substrafl.algorithms.pytorch import TorchNewtonRaphsonAlgo
from substrafl.model_loading import download_algo_state
from fedeca.strategies.fed_avg_decorated import FedAvgDecorated as FedAvg
from fedeca.strategies.newton_raphson_decorated import NewtonRaphsonDecorated as NewtonRaphson

from torch.optim import SGD

from fedeca.algorithms.torch_dp_fed_avg_algo import TorchDPFedAvgAlgo
from fedeca.fedeca_core import LogisticRegressionTorch
from fedeca.utils import (
    Experiment,
    make_accuracy_function,
    make_substrafl_torch_dataset_class,
)
from fedeca.utils.survival_utils import CoxData, make_categorical

if __name__ == "__main__":
    epsilons = [0.1, 1.0, 5.0, 10.0][::-1]
    deltas = [10 ** (-i) for i in range(1, 3)]
    START_SEED = 42
    NDIM = 10
    NUM_ROUNDS = 10
    NUM_UPDATES = 100
    N_REPETITIONS = 5
    BACKEND_TYPE = "subprocess"
    BATCH_SIZE = 32
    na_proportion = 0.0
    seeds = np.arange(START_SEED, START_SEED + N_REPETITIONS).tolist()

    rng = np.random.default_rng(seeds[0])
    # Generating data with strong linear relationship
    simu_coxreg = CoxData(
        n_samples=300,
        ndim=NDIM,
        prop_treated=0.5,
        propensity="linear",
        dtype="float32",
        overlap=100.0,
        seed=rng,
        random_censoring=True,
        censoring_factor=0.3,
        standardize_features=False,
    )
    X, T, C, treated, _ = simu_coxreg.generate_data()
    # Will make first columns to be categorical
    Xcat, Xcont = make_categorical(X, up_to=0)
    # Build the final dataframe using appropriate column names and adding missing values
    cols_dict = {}
    X = np.concatenate((Xcat, Xcont), axis=1)
    for i in range(Xcat.shape[1] + Xcont.shape[1]):
        currentX = X[:, i].astype("float32")
        mask_na = rng.uniform(0, 1, X.shape[0]) > (1.0 - na_proportion)
        currentX[mask_na] = np.nan
        if i < Xcat.shape[1]:
            colname = "cat_col"
        else:
            colname = "col"
            i -= Xcat.shape[1]
        cols_dict[f"{colname}_{i}"] = currentX

        #  The absolute value is superfluous but just to be sure
        cols_dict["T"] = np.abs(T)
        cols_dict["E"] = (1.0 - C).astype("uint8")
        cols_dict["treated"] = treated

        df = pd.DataFrame(cols_dict)
        # Final cast of categorical columns that was impossible due to nan in numpy
        for i in range(Xcat.shape[1]):
            df[f"cat_col_{i}"] = df[f"cat_col_{i}"].astype("Int64")

    results_all_reps = []
    edelta_list = list(product(epsilons, deltas))
    accuracy_metrics_dict = {"accuracy": make_accuracy_function("treated")}
    # We set model and dataloaders to be the same for each rep
    logreg_model = LogisticRegressionTorch(NDIM, torch.float32)
    logreg_dataset_class = make_substrafl_torch_dataset_class(
        ["treated"], "E", "T", dtype="float32", return_torch_tensors=True
    )

    for se in seeds:
        # We run NewtonRaphson wo DP
        class NRAlgo(TorchNewtonRaphsonAlgo):
            """Newton-Raphson algo.

            Parameters
            ----------
            TorchNewtonRaphsonAlgo : _type_
                _description_
            """

            def __init__(self):
                """Instantiate NRAlgo wo DP."""
                super().__init__(
                    model=logreg_model,
                    batch_size=sys.maxsize,
                    criterion=nn.BCELoss(),
                    dataset=logreg_dataset_class,
                    seed=se,  # shouldn't have any effect
                )

        nr_algo = NRAlgo()
        nr_strategy = NewtonRaphson(
            damping_factor=0.8, algo=nr_algo, metric_functions=accuracy_metrics_dict
        )
        regular_xp = Experiment(
            strategies=[nr_strategy],
            num_rounds_list=[10],
        )

        regular_xp.fit(df, nb_clients=3, backend_type=BACKEND_TYPE)
        if regular_xp.ds_client.is_simu:
            final_model = regular_xp.train_data_nodes[0].algo.model
        else:
            final_algo = download_algo_state(
                client=regular_xp.ds_client,
                compute_plan_key=regular_xp.compute_plan_keys[0].key,
                round_idx=None,
            )

            final_model = final_algo.model
        final_pred = (
            final_model(
                torch.from_numpy(
                    df.drop(columns=["treated", "T", "E"]).to_numpy().astype("float32")
                )
            )
            .detach()
            .numpy()
        )
        y_true = df["treated"].to_numpy()
        mean_perf = accuracy_score(y_true, final_pred > 0.5)

        print(f"Mean performance without DP, Perf={mean_perf}")
        results_all_reps.append({"perf": mean_perf, "e": None, "d": None, "seed": se})

        for e, d in edelta_list:
            # We init an algo with the right target epsilon and delta
            # The init (zero init) is the same for all models but batching seeding
            # is controlled by se.
            logreg_model = LogisticRegressionTorch(NDIM, torch.float32)
            optimizer = SGD(logreg_model.parameters(), lr=0.01)

            class DPLogRegAlgo(TorchDPFedAvgAlgo):
                """DP FedAvg algo.

                Parameters
                ----------
                TorchDPFedAvgAlgo : _type_
                    _description_
                """

                def __init__(self):
                    """Instantiate FedAvg algo with DP."""
                    super().__init__(
                        model=logreg_model,
                        criterion=nn.BCELoss(),
                        optimizer=optimizer,
                        dataset=logreg_dataset_class,
                        seed=se,
                        num_updates=NUM_UPDATES,
                        batch_size=BATCH_SIZE,
                        num_rounds=NUM_ROUNDS,
                        dp_target_epsilon=e,
                        dp_target_delta=d,
                        dp_max_grad_norm=1.0,
                    )

            dp_algo = DPLogRegAlgo()
            dp_fedavg_strategy = FedAvg(
                algo=dp_algo, metric_functions=accuracy_metrics_dict
            )
            dp_xp = Experiment(
                strategies=[dp_fedavg_strategy],
                num_rounds_list=[NUM_ROUNDS],
            )
            dp_xp.fit(df, nb_clients=3, backend_type=BACKEND_TYPE)
            if dp_xp.ds_client.is_simu:
                final_model = dp_xp.train_data_nodes[0].algo.model
            else:
                final_algo = download_algo_state(
                    client=dp_xp.ds_client,
                    compute_plan_key=dp_xp.compute_plan_keys[0].key,
                    round_idx=None,
                )
                final_model = final_algo.model
            final_pred = (
                final_model(
                    torch.from_numpy(
                        df.drop(columns=["treated", "T", "E"])
                        .to_numpy()
                        .astype("float32")
                    )
                )
                .detach()
                .numpy()
            )
            y_true = df["treated"].to_numpy()
            mean_perf = accuracy_score(y_true, final_pred > 0.5)

            print(f"Mean performance eps={e}, delta={d}, Perf={mean_perf}")
            # mean_perf = float(np.random.uniform(0, 1.))
            results_all_reps.append({"perf": mean_perf, "e": e, "d": d, "seed": se})

    results = pd.DataFrame.from_dict(results_all_reps)
    results.to_csv("results_logreg_dp_training.csv", index=False)
