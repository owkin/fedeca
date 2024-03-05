"""Utils functions for Substra."""
import logging
import os
import pickle
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Union

import lifelines
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.experiment import execute_experiment, simulate_experiment
from substrafl.nodes import AggregationNode, TestDataNode, TrainDataNode
from substrafl.nodes.schemas import OutputIdentifiers

from fedeca.utils.data_utils import split_dataframe_across_clients

try:
    import git
except ImportError:
    pass
import json
import subprocess

from substrafl.model_loading import (
    FUNCTION_DICT_KEY,
    METADATA_FILE,
    MODEL_DICT_KEY,
    REQUIRED_KEYS,
    _check_environment_compatibility,
)

logger = logging.getLogger(__name__)


class Experiment:
    """Experiment class."""

    def __init__(
        self,
        strategies: list,
        num_rounds_list: list[int],
        ds_client=None,
        train_data_nodes: Union[list[TrainDataNode], None] = None,
        test_data_nodes: Union[list[TestDataNode], None] = None,
        aggregation_node: Union[AggregationNode, None] = None,
        evaluation_frequency: Union[int, None] = None,
        experiment_folder: str = "./experiments",
        clean_models: bool = False,
        fedeca_path: Union[str, None] = None,
        algo_dependencies: Union[list, None] = None,
    ):
        """Initialize an experiment.

        Parameters
        ----------
        ds_client : fl.client.Client
            Federated Learning client object used to register computations.
        strategies : list
            List of strategies to run.
        train_data_nodes : Union[list[TrainDataNode], None]
            List of data nodes for training. If None cannot use the run method
            directly.
        num_rounds_list : list
            List of number of rounds for each strategy.
        test_data_nodes : list, optional
            List of data nodes for testing, by default None.
        aggregation_node : fl.data.DataNode, optional
            Aggregation node, by default None.
        evaluation_frequency : int, optional
            Frequency of evaluation, by default 1.
        experiment_folder : str, optional
            Folder path for experiment outputs, by default "./experiments".
        clean_models : bool, optional
            Whether to clean models after training, by default False.
        fedeca_path : str, optional
            Path to the FedECA package, by default None.
        algo_dependencies : list, optional
            List of algorithm dependencies, by default [].
        """

        assert len(num_rounds_list) == len(strategies)
        self.strategies = strategies
        self.num_rounds_list = num_rounds_list
        self.ds_client = ds_client
        self.train_data_nodes = train_data_nodes
        self.test_data_nodes = test_data_nodes
        self.simu_mode = False

        if self.test_data_nodes is None:
            if self.train_data_nodes is not None:
                self.test_data_nodes = [
                    TestDataNode(
                        t.organization_id,
                        t.data_manager_key,
                        t.data_sample_keys,
                    )
                    for t in self.train_data_nodes
                ]

        self.evaluation_frequency = evaluation_frequency

        self.aggregation_node = aggregation_node
        self.experiment_folder = experiment_folder
        self.clean_models = clean_models

        # Packaging the right dependencies
        if fedeca_path is None:
            fedeca_path = os.getcwd()
        repo_folder = Path(
            git.Repo(fedeca_path, search_parent_directories=True).working_dir
        ).resolve()
        wheel_folder = repo_folder / "temp"
        os.makedirs(wheel_folder, exist_ok=True)
        for stale_wheel in wheel_folder.glob("fedeca*.whl"):
            stale_wheel.unlink()
        process = subprocess.Popen(
            f"python -m build --wheel --outdir {wheel_folder} {repo_folder}",
            shell=True,
            stdout=subprocess.PIPE,
        )
        process.wait()
        assert process.returncode == 0, "Failed to build the wheel"
        wheel_path = next(wheel_folder.glob("fedeca*.whl"))
        if algo_dependencies is None:
            algo_dependencies = []

        self.algo_dependencies = Dependency(
            pypi_dependencies=["numpy==1.23.1", "torch==1.11.0", "lifelines", "pandas"]
            + algo_dependencies,
            local_dependencies=[wheel_path],
        )

        self.experiment_path = str(Path(self.experiment_folder))
        os.makedirs(self.experiment_path, exist_ok=True)
        self.run_strategies = 0
        self.tasks = {}
        self.compute_plan_keys = []
        self.performances_strategies = []

    def fit(
        self,
        data: pd.DataFrame,
        nb_clients: Union[int, None] = None,
        split_method: Union[Callable, str] = "uniform",
        split_method_kwargs: Union[Callable, None] = None,
        data_path: Union[str, None] = None,
        backend_type: str = "subprocess",
        urls: Union[list[str], None] = None,
        tokens: Union[list[str], None] = None,
    ):
        """Fit strategies on global data split across clients.

        For test if provided we use test_data_nodes from int or the
        train_data_nodes in the latter train=test.

        Parameters
        ----------
        data : pd.DataFrame
            The global data to be split has to be a dataframe as we only support
            one opener type.
        nb_clients : Union[int, None], optional
            The number of clients used to split data across, by default None
        split_method : Union[Callable, None], optional
            How to split data across the nb_clients, by default None.
        split_method_kwargs : Union[Callable, None], optional
            Argument of the function used to split data, by default None.
        data_path : Union[str, None]
            Where to store the data on disk when backend is not remote.
        backend_type: str
            The backend to use for substra. Can be either:
            ["subprocess", "docker", "remote"]. Defaults to "subprocess".
        urls: Union[list[str], None]
            Urls corresponding to clients API if using remote backend_type.
            Defaults to None.
        tokens: Union[list[str], None]
            Tokens necessary to authenticate each client API if backend_type
            is remote. Defauts to None.
        """
        # Reset experiment so that it can fit on a new dataset
        self.reset_experiment()

        if data_path is not None:
            self.experiment_path = data_path

        # We first have to create the TrainDataNodes objects for this we split
        # the data into nb_clients using split_method
        (
            self.clients,
            self.train_data_nodes,
            test_data_nodes,
            _,
            _,
        ) = split_dataframe_across_clients(
            df=data,
            n_clients=nb_clients,
            split_method=split_method,
            split_method_kwargs=split_method_kwargs,
            backend_type=backend_type,
            data_path=data_path,
            urls=urls,
            tokens=tokens,
        )
        if self.test_data_nodes is None:
            self.test_data_nodes = test_data_nodes
        self.run()

    def run(self, num_strategies_to_run=None):
        """Run the experiment.

        Parameters
        ----------
        num_strategies_to_run : int, optional
            Number of strategies to run, by default None.
        """
        assert (
            self.train_data_nodes is not None
        ), "you have to define train_data_nodes first before running"
        assert (
            self.test_data_nodes is not None
        ), "you have to define test_data_nodes first before running"
        if num_strategies_to_run is None:
            num_strategies_to_run = len(self.strategies) - self.run_strategies
        assert (self.run_strategies + num_strategies_to_run) <= len(
            self.strategies
        ), f"""You cannot run {num_strategies_to_run} strategies more there is only
        {len(self.strategies)} strategies and you have already run {self.run_strategies}
        of them."""
        # If no client is given we take the first one
        if self.ds_client is None:
            self.ds_client = self.clients[list(self.clients.keys())[0]]

        # If no AggregationNode is given we take the first one
        if self.aggregation_node is None:
            logger.info("Using the first client as a server.")
            kwargs_agg_node = {
                "organization_id": self.train_data_nodes[0].organization_id
            }
            self.aggregation_node = AggregationNode(**kwargs_agg_node)

        if not hasattr(self, "experiment_kwargs"):
            self.experiment_kwargs = {
                "experiment_folder": self.experiment_path,
                "clean_models": self.clean_models,
                "dependencies": self.algo_dependencies,
                "client": self.ds_client,
            }
        if hasattr(self.ds_client, "is_simu"):
            self.simu_mode = self.ds_client.is_simu

        # inelegant but cannot slice on a zip object
        strategies = self.strategies[
            self.run_strategies : (self.run_strategies + num_strategies_to_run)
        ]  # noqa: E203
        num_rounds_list = self.num_rounds_list[
            self.run_strategies : (
                self.run_strategies + num_strategies_to_run
            )  # noqa: E203
        ]
        for i, (strategy, num_rounds) in enumerate(zip(strategies, num_rounds_list)):

            current_kwargs = self.experiment_kwargs
            current_kwargs["strategy"] = strategy
            current_kwargs["num_rounds"] = num_rounds
            current_kwargs["train_data_nodes"] = self.train_data_nodes
            current_kwargs["aggregation_node"] = self.aggregation_node
            # Evaluation frequency depend on current strategy
            # If None evaluate once at the end of the strategy
            if self.evaluation_frequency is None:
                evaluation_strategy = EvaluationStrategy(
                    test_data_nodes=self.test_data_nodes,
                    eval_rounds=[num_rounds_list[i]],
                )
            else:
                evaluation_strategy = EvaluationStrategy(
                    test_data_nodes=self.test_data_nodes,
                    eval_frequency=self.evaluation_frequency[i],
                )
            current_kwargs["evaluation_strategy"] = evaluation_strategy
            current_kwargs["name"] = f"Fedeca: {strategy.__class__.__name__}"

            if not self.simu_mode:
                xp_output = execute_experiment(**current_kwargs)
            else:
                (
                    scores,
                    intermediate_state_train,
                    intermediate_state_agg,
                ) = simulate_experiment(**current_kwargs)
                xp_output = scores

                robust_cox_variance = False
                for idx, s in enumerate(list(scores.values())):
                    logger.info(f"====Client {idx}====")
                    try:
                        logger.info(s[-1])
                    except IndexError:
                        robust_cox_variance = True
                        logger.info("No metric")
                # TODO Check that it is well formatted it's probably not
                self.performances_strategies.append(pd.DataFrame(xp_output))
                # Hacky hacky hack
                if robust_cox_variance:
                    xp_output = self.train_data_nodes
                else:
                    xp_output = self.train_data_nodes[0]

            self.compute_plan_keys.append(xp_output)

            if not (self.simu_mode):
                self.tasks[self.compute_plan_keys[i].key] = {}
                tasks = self.ds_client.list_task(
                    filters={"compute_plan_key": [self.compute_plan_keys[i].key]}
                )[::-1]
                tasks_names = [t.function.name for t in tasks]
                self.tasks[self.compute_plan_keys[i].key]["tasks"] = tasks
                self.tasks[self.compute_plan_keys[i].key]["tasks_names"] = tasks_names
                self.tasks[self.compute_plan_keys[i].key]["num_tasks"] = len(tasks)

            self.run_strategies += 1

    def get_outmodel(self, task_name, strategy_idx=0, idx_task=0):
        """Get the output model.

        Parameters
        ----------
        task_name : str
            Name of the task.
        strategy_idx : int, optional
            Index of the strategy, by default 0.
        idx_task : int, optional
            Index of the task, by default 0.
        """
        assert not (self.simu_mode), "This function cannot be used in simu mode"

        # We get all matches and order them chronologically
        tasks_dict_from_strategy = self.tasks[self.compute_plan_keys[strategy_idx].key]
        return get_outmodel_function(
            task_name, idx_task=idx_task, tasks_dict=tasks_dict_from_strategy
        )

    def reset_experiment(self):
        """Reset the state of the object.

        So it can be fit with a new dataset.
        """
        self.run_strategies = 0
        self.tasks = {}
        self.compute_plan_keys = []
        self.performances_strategies = []
        self.train_data_nodes = None
        self.test_data_nodes = None


def get_outmodel_function(
    task_name, client, compute_plan_key=None, idx_task=0, tasks_dict={}
):
    """Retrieve an output model from a task or tasks_dict."""
    assert (
        compute_plan_key is not None or tasks_dict
    ), "Please provide a tasks dict or a compute plan key"
    if tasks_dict:
        assert compute_plan_key is None
        assert (
            ("num_tasks" in tasks_dict)
            and ("tasks" in tasks_dict)
            and ("tasks_names" in tasks_dict)
        )
    else:
        assert isinstance(compute_plan_key, str)
        assert not tasks_dict
        assert client is not None
        tasks = client.list_task(filters={"compute_plan_key": [compute_plan_key]})[::-1]
        tasks_names = [t.function.name for t in tasks]
        tasks_dict = {}
        tasks_dict["tasks"] = tasks
        tasks_dict["tasks_names"] = tasks_names
        tasks_dict["num_tasks"] = len(tasks)

    num_tasks = tasks_dict["num_tasks"]
    compatible_indices = [
        i for i in range(num_tasks) if tasks_dict["tasks_names"][i] == task_name
    ]
    idx_outmodel = compatible_indices[idx_task]
    outmodel_task = tasks_dict["tasks"][idx_outmodel]
    with tempfile.TemporaryDirectory() as temp_dir:
        identifier = outmodel_task.function.outputs[0].identifier
        model_path = client.download_model_from_task(
            outmodel_task.key, identifier, temp_dir
        )
        if identifier in ["model", "shared"]:
            # Assumes it can be unserialized with pickle
            with open(model_path, "rb") as f:
                outmodel = pickle.load(f)
        elif identifier == "local":
            # Assumes it can be unserialized with torch
            outmodel = torch.load(model_path)
        else:
            raise ValueError(f"Identifier {identifier} not recognized")

    return outmodel


class SubstraflTorchDataset(torch.utils.data.Dataset):
    """Substra toch dataset class."""

    def __init__(
        self,
        data_from_opener,
        is_inference: bool,
        target_columns=["T", "E"],
        columns_to_drop=[],
        dtype="float64",
        return_torch_tensors=False,
    ):
        """Initialize SubstraflTorchDataset class.

        Parameters
        ----------
        data_from_opener : pandas.DataFrame
            Data samples.
        is_inference : bool
            Flag indicating if the dataset is for inference.
        target_columns : list, optional
            List of target columns, by default ["T", "E"].
        columns_to_drop : list, optional
            List of columns to drop, by default [].
        dtype : str, optional
            Data type, by default "float64".
        return_torch_tensors: bool, optional
            Returns torch.Tensor, actually substra generally expects your dataset
            to return torch.Tensor and not numpy as the training loop uses pytorch
            and doesn't explicitly call torch.from_numpy. This is different from
            say NewtonRaphson and WebDisco which are numpy-based. Defaults to False.
        """
        self.data = data_from_opener
        self.is_inference = is_inference
        self.target_columns = target_columns
        self.columns_to_drop = list(set(columns_to_drop + self.target_columns))
        self.x = self.data.drop(columns=self.columns_to_drop).to_numpy().astype(dtype)
        self.y = self.data[self.target_columns].to_numpy().astype(dtype)
        self.return_torch_tensors = return_torch_tensors

    def __getitem__(self, idx):
        """Get item."""
        if self.is_inference:
            x = self.x[idx]
            if self.return_torch_tensors:
                x = torch.from_numpy(x)
            return x

        else:
            x, y = self.x[idx], self.y[idx]
            if self.return_torch_tensors:
                x, y = torch.from_numpy(x), torch.from_numpy(y)
            return x, y

    def __len__(self):
        """Get length."""
        return len(self.data.index)


def make_substrafl_torch_dataset_class(
    target_cols,
    event_col,
    duration_col,
    dtype="float64",
    return_torch_tensors=False,
):
    """Create a custom SubstraflTorchDataset class for survival analysis.

    Parameters
    ----------
    target_cols : list
        List of target columns.
    event_col : str
        Name of the event column.
    duration_col : str
        Name of the duration column.
    dtype : str, optional
        Data type, by default "float64".
    return_torch_tensors : bool, optional
        Returns torch.Tensor. Defaults to False.

    Returns
    -------
    type
        Custom SubstraflTorchDataset class.
    """
    assert len(target_cols) == 1 or all(
        [t in [event_col, duration_col] for t in target_cols]
    )
    if len(target_cols) == 1:
        logger.info(
            f"Making a dataset class to fit a model to predict {target_cols[0]}"
        )
        columns_to_drop = [event_col, duration_col]
    elif len(target_cols) == 2:
        assert set(target_cols) == set(
            [event_col, duration_col]
        ), "Your targets should be event_col and duration_col"
        # DO NOT MODIFY THIS LINE !!!!!
        target_cols = [duration_col, event_col]
        columns_to_drop = []

    class MySubstraflTorchDataset(SubstraflTorchDataset):
        def __init__(self, data_from_opener, is_inference):
            super().__init__(
                data_from_opener=data_from_opener,
                is_inference=is_inference,
                target_columns=target_cols,
                columns_to_drop=columns_to_drop,
                dtype=dtype,
                return_torch_tensors=return_torch_tensors,
            )

    return MySubstraflTorchDataset


def make_c_index_function(duration_col: str, event_col: str):
    """Build C-index function.

    Parameters
    ----------
    duration_col : str,
        Column name for the duration.
    event_col : str,
        Column name for the event.
    """

    def c_index(data_from_opener, predictions):
        times_true = data_from_opener[duration_col]
        events = data_from_opener[event_col]

        y_pred = predictions

        c_index = lifelines.utils.concordance_index(times_true, -y_pred, events)
        return c_index

    return c_index


def make_accuracy_function(treatment_col: str):
    """Build accuracy function.

    Parameters
    ----------
    treatment_col: str,
        Column name for the treatment allocation.
    """

    def accuracy(data_from_opener, predictions):
        y_true = data_from_opener[treatment_col]
        y_pred = predictions
        return accuracy_score(y_true, y_pred > 0.5)

    return accuracy


def download_train_task_models_by_round(
    client, dest_folder, compute_plan_key, round_idx
):
    """Download models associated with a specific round of a train task."""
    compute_plan = client.get_compute_plan(compute_plan_key)

    _check_environment_compatibility(metadata=compute_plan.metadata)

    folder = Path(dest_folder)
    folder.mkdir(exist_ok=True, parents=True)

    if round_idx is None:
        round_idx = compute_plan.metadata["num_rounds"]

    # Retrieve local train task key
    local_train_tasks = client.list_task(
        filters={
            "compute_plan_key": [compute_plan.key],
            "metadata": [{"key": "round_idx", "type": "is", "value": str(round_idx)}],
            "worker": [client.organization_info().organization_id],
        }
    )
    local_train_tasks = [t for t in local_train_tasks if t.tag == "train"]
    train_task = local_train_tasks[-1]

    # Get the associated head model (local state)
    model_file = client.download_model_from_task(
        train_task.key, folder=folder, identifier=OutputIdentifiers.local
    )
    function_file = client.download_function(
        train_task.function.key, destination_folder=folder
    )

    # Environment requirements and local state path
    metadata = {k: v for k, v in compute_plan.metadata.items() if k in REQUIRED_KEYS}
    metadata[MODEL_DICT_KEY] = str(model_file.relative_to(folder))
    metadata[FUNCTION_DICT_KEY] = str(function_file.relative_to(folder))
    metadata_path = folder / METADATA_FILE
    metadata_path.write_text(json.dumps(metadata))
    return model_file
