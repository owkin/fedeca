"""Tests for efficiently bootstrapping FL strategies."""
import numpy as np
import pytest
from pathlib import Path
import os
from itertools import product
from substrafl.experiment import execute_experiment
from substrafl.dependency import Dependency
from substrafl.algorithms.pytorch import TorchFedAvgAlgo, TorchNewtonRaphsonAlgo
from substrafl.index_generator import NpIndexGenerator
import torch
from substrafl.strategies import FedAvg, NewtonRaphson
from fedeca.utils.survival_utils import CoxData
from fedeca import LogisticRegressionTorch
from fedeca.utils import (
    make_accuracy_function,
    make_substrafl_torch_dataset_class,
)
from substrafl.strategies.strategy import Strategy
from fedeca.strategies import make_bootstrap_metric_function, make_bootstrap_strategy
from fedeca.utils.data_utils import split_dataframe_across_clients
from substrafl.nodes import AggregationNode
from substrafl.model_loading import download_algo_state
import shutil


logreg_dataset_class = make_substrafl_torch_dataset_class(
            ["treatment"],
            event_col="event",
            duration_col="time",
            return_torch_tensors=True,
        )

accuracy = make_accuracy_function("treatment")

class UnifLogReg(LogisticRegressionTorch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    def __init__(self):
        super().__init__(
            model=logreg_model,
            criterion=criterion,
            batch_size=BATCH_SIZE,
            dataset=logreg_dataset_class,
            seed=SEED,
            use_gpu=False,
        )

list_strategy_params = [{"strategy": {"strategy_class": FedAvg, "strategy_kwargs": {"algo": TorchLogRegFedAvgAlgo()}}, "get_true_nb_rounds": lambda x : x}, 
                   {"strategy": {"strategy_class": NewtonRaphson, "strategy_kwargs": {"algo": TorchLogRegNRAlgo()}}, "get_true_nb_rounds": lambda x : x}]


@pytest.mark.parametrize("strategy_params, num_rounds", product(list_strategy_params, range(5)))
def test_bootstrapping(strategy_params: dict, num_rounds: int):
    """Tests of data generation with constant cate."""
    # Let's generate 1000 data samples with 10 covariates
    data = CoxData(seed=42, n_samples=1000, ndim=50, overlap=10., propensity="linear")
    df = data.generate_dataframe()

    # We remove the true propensity score
    df = df.drop(columns=["propensity_scores"], axis=1)

    
    bootstrap_seeds_list = [42, 43, 44]
    n_bootstrap_samples = len(bootstrap_seeds_list)
    strategy = strategy_params["strategy"]["strategy_class"](**strategy_params["strategy"]["strategy_kwargs"])
    #btst_accuracy = make_bootstrap_metric_function(accuracy, n_bootstrap_samples=n_bootstrap_samples)
    btst_strategy = make_bootstrap_strategy(strategy, bootstrap_seeds_list=bootstrap_seeds_list)

    # inefficient bootstrap
    bootstrapped_models = []
    for idx, seed in enumerate(bootstrap_seeds_list):
        rng = np.random.default_rng(seed)
        bootstrapped_df = df.sample(df.shape[0], replace=True, random_state=rng)
        clients, train_data_nodes, _, _, _ = split_dataframe_across_clients(
            bootstrapped_df,
            n_clients=N_CLIENTS,
            split_method= "uniform",
            split_method_kwargs=None,
            data_path="./data",
            backend_type="subprocess",
        )
        first_key = list(clients.keys())[0]
        my_eval_strategy = None
        aggregation_node = AggregationNode(clients[first_key].organization_info().organization_id)

        xp_dir = str(Path.cwd() / "tmp" / "experiment_summaries")
        os.makedirs(xp_dir, exist_ok=True)
        dependencies = Dependency(pypi_dependencies=["numpy==1.24.3", "scikit-learn==1.3.1", "torch==2.0.1", "--extra-index-url https://download.pytorch.org/whl/cpu"])
        compute_plan = execute_experiment(
            client=clients[first_key],
            strategy=strategy,
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
        bootstrapped_models.append(algo.model)
        # Clean up
        shutil.rmtree('./data')
    
    # efficient bootstrap

    clients, train_data_nodes, _, _, _ = split_dataframe_across_clients(
        df,
        n_clients=N_CLIENTS,
        split_method= "uniform",
        split_method_kwargs=None,
        data_path="./data",
        backend_type="subprocess",
    )
    first_key = list(clients.keys())[0]
    my_eval_strategy = None
    aggregation_node = AggregationNode(clients[first_key].organization_info().organization_id)

    xp_dir = str(Path.cwd() / "tmp" / "experiment_summaries")
    os.makedirs(xp_dir, exist_ok=True)
    dependencies = Dependency(pypi_dependencies=["numpy==1.24.3", "scikit-learn==1.3.1", "torch==2.0.1", "--extra-index-url https://download.pytorch.org/whl/cpu"])
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
    bootstrapped_models = algo.model







