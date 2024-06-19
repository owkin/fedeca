"""Tests for efficiently bootstrapping FL strategies."""
import copy
import os
import shutil
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from substrafl.algorithms.pytorch import TorchFedAvgAlgo, TorchNewtonRaphsonAlgo
from substrafl.dependency import Dependency
from substrafl.experiment import execute_experiment
from substrafl.index_generator import NpIndexGenerator
from substrafl.model_loading import download_algo_state
from substrafl.nodes import AggregationNode
from substrafl.strategies import FedAvg, NewtonRaphson

from fedeca import FedECA, LogisticRegressionTorch
from fedeca.algorithms import TorchWebDiscoAlgo
from fedeca.strategies import WebDisco, make_bootstrap_strategy
from fedeca.utils import make_accuracy_function, make_substrafl_torch_dataset_class
from fedeca.utils.data_utils import split_dataframe_across_clients, uniform_split
from fedeca.utils.survival_utils import CoxData, CoxPHModelTorch

logreg_dataset_class = make_substrafl_torch_dataset_class(
    ["treatment"],
    event_col="event",
    duration_col="time",
    return_torch_tensors=True,
)
survival_dataset_class = make_substrafl_torch_dataset_class(
    ["time", "event"],
    "event",
    "time",
    dtype="float64",
)
accuracy = make_accuracy_function("treatment")


class UnifLogReg(LogisticRegressionTorch):
    """Spawns FedECA logreg model with uniform weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.manual_seed(42)
        self.fc1.weight.data.uniform_(-1, 1)


logreg_model = UnifLogReg(ndim=50)

optimizer = torch.optim.Adam(logreg_model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()
BATCH_SIZE = 32
NUM_UPDATES = 10
SEED = 42
N_CLIENTS = 2

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES,
)


class TorchLogRegFedAvgAlgo(TorchFedAvgAlgo):
    """Spawns FedAvg algo with logreg model with uniform weights."""

    def __init__(self):
        super().__init__(
            model=logreg_model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=logreg_dataset_class,
            seed=SEED,
            use_gpu=False,
        )


class TorchLogRegNRAlgo(TorchNewtonRaphsonAlgo):
    """Spawns NR algo with logreg model with uniform weights."""

    def __init__(self):
        super().__init__(
            model=logreg_model,
            criterion=criterion,
            batch_size=BATCH_SIZE,
            dataset=logreg_dataset_class,
            seed=SEED,
            use_gpu=False,
        )


cox_model = CoxPHModelTorch(ndim=1, torch_dtype=torch.float64)


class WDAlgo(TorchWebDiscoAlgo):
    """Spawns WebDisco algo with cox model."""

    def __init__(self, propensity_model, robust):
        super().__init__(
            model=cox_model,
            # TODO make this batch-size argument disappear from
            # webdisco algo
            batch_size=sys.maxsize,
            dataset=survival_dataset_class,
            seed=SEED,
            duration_col="time",
            event_col="event",
            treated_col="treatment",
            standardize_data=True,
            penalizer=0.0,
            l1_ratio=0.1,
            initial_step_size=1.0,
            learning_rate_strategy="lifelines",
            store_hessian=True,
            propensity_model=propensity_model,
            propensity_strategy="iptw",
            robust=robust,
        )
        self._propensity_model = propensity_model


list_strategy_params = [
    {
        "strategy": {
            "strategy_class": FedAvg,
            "strategy_kwargs": {"algo": TorchLogRegFedAvgAlgo()},
        },
        "get_true_nb_rounds": lambda x: x,
    },
    {
        "strategy": {
            "strategy_class": NewtonRaphson,
            "strategy_kwargs": {"algo": TorchLogRegNRAlgo(), "damping_factor": 0.1},
        },
        "get_true_nb_rounds": lambda x: x,
    },
    {
        "strategy": {
            "strategy_class": WebDisco,
            "strategy_kwargs": {
                "algo": WDAlgo(propensity_model=logreg_model, robust=True),
                "metric_functions": {"accuracy": accuracy},
                "standardize_data": True,
            },
        },
        "get_true_nb_rounds": lambda x: x,
    },
]


@pytest.mark.parametrize(
    "strategy_params, num_rounds", product(list_strategy_params, range(1, 2))
)
def test_bootstrapping_per_client(strategy_params: dict, num_rounds: int):
    """Tests of data generation with constant cate."""
    # Let's generate 1000 data samples with 10 covariates
    data = CoxData(seed=42, n_samples=1000, ndim=50, overlap=10.0, propensity="linear")
    original_df = data.generate_dataframe()

    # We remove the true propensity score
    original_df = original_df.drop(columns=["propensity_scores"], axis=1)

    bootstrap_seeds_list = [42, 43, 44]

    strategy = strategy_params["strategy"]["strategy_class"](
        **strategy_params["strategy"]["strategy_kwargs"]
    )

    btst_strategy, _ = make_bootstrap_strategy(
        strategy,
        bootstrap_seeds=bootstrap_seeds_list,
    )

    # inefficient bootstrap
    splits = {}
    bootstrapped_models_gt = []

    for idx, seed in enumerate(bootstrap_seeds_list):
        # We need to mimic the bootstrap sampling of the data which is per-client
        clients_indices_list = uniform_split(original_df, N_CLIENTS)
        dfs = [original_df.iloc[clients_indices_list[i]] for i in range(N_CLIENTS)]
        # Rng needs to be defined within the loop so that it's not updated by the
        # sampling
        dfs = [
            df.sample(
                df.shape[0], replace=True, random_state=np.random.default_rng(seed)
            )
            for df in dfs
        ]

        size_dfs = [len(df.index) for df in dfs]
        bootstrapped_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

        def trivial_split(df, n_clients):
            start = 0
            end = 0
            clients_indices_list = []
            for i in range(n_clients):
                end += size_dfs[i]
                clients_indices_list.append(range(start, end))
                start = end
            return clients_indices_list

        clients, train_data_nodes, _, current_dfs, _ = split_dataframe_across_clients(
            bootstrapped_df,
            n_clients=N_CLIENTS,
            split_method=trivial_split,
            split_method_kwargs=None,
            data_path="./data",
            backend_type="subprocess",
        )
        splits[seed] = current_dfs

        first_key = list(clients.keys())[0]
        my_eval_strategy = None
        aggregation_node = AggregationNode(
            clients[first_key].organization_info().organization_id
        )

        xp_dir = str(Path.cwd() / "tmp" / "experiment_summaries")
        os.makedirs(xp_dir, exist_ok=True)
        dependencies = Dependency(
            pypi_dependencies=[
                "numpy==1.24.3",
                "scikit-learn==1.3.1",
                "torch==2.0.1",
                "--extra-index-url https://download.pytorch.org/whl/cpu",
            ]
        )

        current_strategy = copy.deepcopy(strategy)

        print(f"Bootstrap {idx}")
        compute_plan = execute_experiment(
            client=clients[first_key],
            strategy=current_strategy,
            train_data_nodes=train_data_nodes,
            evaluation_strategy=my_eval_strategy,
            aggregation_node=aggregation_node,
            num_rounds=num_rounds,
            experiment_folder=xp_dir,
            dependencies=dependencies,
            clean_models=False,
            name=f"FedECA-boostrap{idx}",
        )

        algo = download_algo_state(
            client=clients[first_key],
            compute_plan_key=compute_plan.key,
            round_idx=strategy_params["get_true_nb_rounds"](num_rounds),
        )
        bootstrapped_models_gt.append(copy.deepcopy(algo.model))
        # Clean up
        shutil.rmtree("./data")

    # efficient bootstrap
    clients, train_data_nodes, _, efficient_dfs, _ = split_dataframe_across_clients(
        original_df,
        n_clients=N_CLIENTS,
        split_method="uniform",
        split_method_kwargs=None,
        data_path="./data",
        backend_type="subprocess",
    )
    eff_splits = {}
    for seed in bootstrap_seeds_list:
        eff_splits[seed] = [
            eff_df.sample(
                eff_df.shape[0], replace=True, random_state=np.random.default_rng(seed)
            )
            for eff_df in efficient_dfs
        ]
    for seed in bootstrap_seeds_list:
        assert all(
            [
                np.allclose(np.asarray(l1), np.asarray(l2))
                for l1, l2 in zip(eff_splits[seed], splits[seed])
            ]
        )

    first_key = list(clients.keys())[0]
    my_eval_strategy = None
    aggregation_node = AggregationNode(
        clients[first_key].organization_info().organization_id
    )

    xp_dir = str(Path.cwd() / "tmp" / "experiment_summaries")
    os.makedirs(xp_dir, exist_ok=True)
    dependencies = Dependency(
        pypi_dependencies=[
            "numpy==1.24.3",
            "scikit-learn==1.3.1",
            "torch==2.0.1",
            "--extra-index-url https://download.pytorch.org/whl/cpu",
        ]
    )

    current_strategy = copy.deepcopy(strategy)

    btst_strategy, _ = make_bootstrap_strategy(
        current_strategy,
        bootstrap_seeds=bootstrap_seeds_list,
    )
    print("Efficient Bootstrap")
    compute_plan = execute_experiment(
        client=clients[first_key],
        strategy=btst_strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        experiment_folder=xp_dir,
        dependencies=dependencies,
        clean_models=False,
        name="FedECA-all-bootstraps",
    )

    algo = download_algo_state(
        client=clients[first_key],
        compute_plan_key=compute_plan.key,
        round_idx=strategy_params["get_true_nb_rounds"](num_rounds),
    )
    # We don't care about the first model which is not bootstrapped
    bootstrapped_models_efficient = [alg._model for alg in algo.individual_algos][1:]

    for model1, model2 in zip(bootstrapped_models_gt, bootstrapped_models_efficient):
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                assert False
    # Clean up
    shutil.rmtree("./data")


def test_bootstrapping_per_client_end2end():
    """Tests of data generation with constant cate."""
    # Let's generate 1000 data samples with 10 covariates
    data = CoxData(seed=42, n_samples=1000, ndim=50, overlap=10.0, propensity="linear")
    original_df = data.generate_dataframe()

    # We remove the true propensity score
    original_df = original_df.drop(columns=["propensity_scores"], axis=1)

    bootstrap_seeds_list = [42, 43, 44, 44]

    # inefficient bootstrap
    splits = {}
    btst_results = []

    for idx, seed in enumerate(bootstrap_seeds_list):
        # We need to mimic the bootstrap sampling of the data which is per-client
        clients_indices_list = uniform_split(original_df, N_CLIENTS)
        dfs = [original_df.iloc[clients_indices_list[i]] for i in range(N_CLIENTS)]
        # Rng needs to be defined within the loop so that it's not updated by the
        # sampling
        dfs = [
            df.sample(
                df.shape[0], replace=True, random_state=np.random.default_rng(seed)
            )
            for df in dfs
        ]

        size_dfs = [len(df.index) for df in dfs]
        bootstrapped_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

        def trivial_split(df, n_clients):
            start = 0
            end = 0
            clients_indices_list = []
            for i in range(n_clients):
                end += size_dfs[i]
                clients_indices_list.append(range(start, end))
                start = end
            return clients_indices_list

        clients, train_data_nodes, _, current_dfs, _ = split_dataframe_across_clients(
            bootstrapped_df,
            n_clients=N_CLIENTS,
            split_method=trivial_split,
            split_method_kwargs=None,
            data_path="./data",
            backend_type="subprocess",
        )
        splits[seed] = current_dfs

        first_key = list(clients.keys())[0]
        xp_dir = str(Path.cwd() / "tmp" / "experiment_summaries")
        os.makedirs(xp_dir, exist_ok=True)
        fed_iptw = FedECA(
            ndim=50,
            ds_client=clients[first_key],
            train_data_nodes=train_data_nodes,
            treated_col="treatment",
            duration_col="time",
            event_col="event",
            variance_method="naive",
            num_rounds_list=[2, 3],
        )
        print(f"Bootstrap {idx}")
        fed_iptw.run()
        btst_results.append(fed_iptw.results_)
        # Clean up
        shutil.rmtree("./data")

    # Variance estimation of beta through inefficient bootstrap
    beta_inefficient = pd.concat(
        [btst_res["coef"] for btst_res in btst_results], axis=0
    )

    # efficient bootstrap
    clients, train_data_nodes, _, _, _ = split_dataframe_across_clients(
        original_df,
        n_clients=N_CLIENTS,
        split_method="uniform",
        split_method_kwargs=None,
        data_path="./data",
        backend_type="subprocess",
    )

    first_key = list(clients.keys())[0]
    fed_iptw = FedECA(
        ndim=50,
        ds_client=clients[first_key],
        train_data_nodes=train_data_nodes,
        treated_col="treatment",
        duration_col="time",
        event_col="event",
        variance_method="bootstrap",
        bootstrap_seeds=bootstrap_seeds_list,
        num_rounds_list=[2, 3],
    )
    print("Efficient bootstrap")
    fed_iptw.run()
    # We don't care about the first run which is not bootstrapped
    beta_efficient = fed_iptw.final_params_array.flatten()[1:]

    assert np.allclose(beta_inefficient.to_numpy(), beta_efficient.flatten())


def test_global_bootstrapping_indices():
    """Tests of data generation with constant cate."""
    # Let's generate 1000 data samples with 10 covariates
    data = CoxData(seed=42, n_samples=1000, ndim=50, overlap=10.0, propensity="linear")
    original_df = data.generate_dataframe()

    # We remove the true propensity score
    original_df = original_df.drop(columns=["propensity_scores"], axis=1)
    clients_indices_list = uniform_split(original_df, N_CLIENTS)
    dfs = [original_df.iloc[clients_indices_list[i]] for i in range(N_CLIENTS)]
    for idx_c, df in enumerate(dfs):
        df["center"] = "client_" + str(idx_c)
    # Here we would enter the length of the total dataset
    TOTAL_LENGTH = len(original_df.index)
    indices = np.arange(TOTAL_LENGTH)
    np.random.seed(42)
    # np.random.shuffle(indices)
    bootstrap_seeds_list = [42, 43, 44, 44]
    global_btst_indices = {}
    per_client_btst_indices = {}
    for seed in bootstrap_seeds_list:
        per_client_btst_indices[seed] = {}
        global_indices_list_per_client = [[] for _ in range(N_CLIENTS)]
        # Avoid edge cases of bootstrap where some clients might not have any data
        # or not enough data to compute a variance
        while not all(
            [
                len(global_indices) >= 2
                for global_indices in global_indices_list_per_client
            ]
        ):
            global_indices_list = np.random.choice(
                indices, size=TOTAL_LENGTH, replace=True
            )
            global_indices_list_per_client = []
            for client_indices in clients_indices_list:
                global_indices_list_per_client.append(
                    [
                        global_idx
                        for global_idx in global_indices_list
                        if global_idx in client_indices
                    ]
                )
        global_btst_indices[seed] = global_indices_list

        # Now we "just" need to translate global_indices in per-client indices
        for idx_c, client_indices in enumerate(clients_indices_list):
            per_client_btst_indices[seed]["client_" + str(idx_c)] = [
                clients_indices_list[idx_c].index(global_idx)
                for global_idx in global_indices_list_per_client[idx_c]
            ]
            assert original_df.iloc[global_indices_list_per_client[idx_c]].equals(
                dfs[idx_c]
                .iloc[per_client_btst_indices[seed]["client_" + str(idx_c)]]
                .drop(columns=["center"])
            )

    def global_bootstrap(data, seed):
        assert data["center"].nunique() == 1
        center = data["center"].unique()[0]
        data = data.drop(columns=["center"])
        indices_center = per_client_btst_indices[seed][center]
        return data.iloc[indices_center]

    for seed in bootstrap_seeds_list:
        global_indices_list_per_client = []
        for client_indices in clients_indices_list:
            global_indices_list_per_client.append(
                [
                    global_idx
                    for global_idx in global_btst_indices[seed]
                    if global_idx in client_indices
                ]
            )
        for client in range(N_CLIENTS):
            assert global_bootstrap(dfs[client], seed).equals(
                original_df.iloc[global_indices_list_per_client[client]]
            )
