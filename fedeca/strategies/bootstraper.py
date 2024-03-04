"""Bootstrap substra strategy in an efficient fashion."""
import copy
import inspect
import os
import re
import tempfile
import types
import zipfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.remote import remote, remote_data
from substrafl.strategies.strategy import Strategy

from fedeca.utils.survival_utils import BootstrapMixin


def make_bootstrap_strategy(
    strategy: Strategy,
    n_bootstrap: Union[int, None] = None,
    bootstrap_seeds: Union[list[int], None] = None,
    bootstrap_function: Union[Callable, None] = None,
):
    """Bootstrap a substrafl strategy wo impacting the number of compute tasks.

    In order to reduce the bottleneck of substra when bootstraping a strategy
    we need to go over the strategy compute plan and modifies each local atomic
    task to execute n_bootstrap times on bootstraped data. Each modified task
    returns a list of n_bootstraps original outputs obtained on each bootstrap.
    Each aggregation task is then modified to aggregate the n_bootstrap outputs
    independently.
    This code heavily uses some code patterns invented by Arthur Pignet.

    Parameters
    ----------
    strategy : Strategy
        The strategy to bootstrap.
    n_bootstrap : Union[int, None]
        Number of bootstrap to be performed. If None will use
        len(bootstrap_seeds) instead. If bootstrap_seeds is given
        seeds those seeds will be used for the generation
        otherwise seeds are generated randomly.
    bootstrap_seeds : Union[list[int], None]
        The list of seeds used for bootstrapping random states.
        If None will generate n_bootstrap randomly, in the presence
        of both allways use bootstrap_seeds.
    bootstrap_function : Union[Callable, None]
        A function with signature f(datasamples, seed) that returns a bootstrapped
        version of the data.
        If None, use the BootstrapMixin function.
        Note that this can be used to provide splits/cross-validation capabilities
        as well where seed would be the fold number in a flattened list of folds.

    Returns
    -------
    Strategy
        The resulting efficiently bootstrapped strategy
    """
    # We dynamically get all methods from strategy and algo except 'magic'
    # methods that have dunderscores
    orig_strat_methods_names = [
        method_name
        for method_name in dir(strategy)
        if callable(getattr(strategy, method_name))
        and not re.match(r"__.+__", method_name)
    ]
    orig_algo_methods_names = [
        method_name
        for method_name in dir(strategy.algo)
        if callable(getattr(strategy.algo, method_name))
        and not re.match(r"__.+__", method_name)
    ]

    # We need to do two things 1. filter-out inner methods, that is to say
    # methods that don't get diretly decorated and called by substrafl with
    # remote/remote_data but are called themselves by functions that do.
    # Since substrafl is clean, methods decorated by substrafl have a __wrapped__
    # attribute.
    # 2. differentiate between aggregations and local computations
    # We'll use the signature of the methods to accomplish 2.
    # Note that there is no way to know the order in which they will be applied
    # in the compute plan.
    # We are just identifying aggregations and local computations here.

    local_functions_names = {"algo": [], "strategy": []}
    aggregations_names = {"algo": [], "strategy": []}
    for idx, method_name in enumerate(
        orig_strat_methods_names + orig_algo_methods_names
    ):
        if idx < len(orig_strat_methods_names):
            obj = strategy
            key = "strategy"
        else:
            obj = strategy.algo
            key = "algo"

        # The second condition is used as we deal separately with save_local_state
        # and load_local_state
        method_args_dict = inspect.signature(getattr(obj, method_name)).parameters
        if not (
            ("shared_states" in method_args_dict)
            or ("shared_state" in method_args_dict)
        ) or (method_name in ["save_local_state", "load_local_state"]):
            continue
        # We are dealing with the predict method this is also a method
        # we need to change but as it's common for all strategies we will
        # handle it separately later.
        if "predictions_path" in method_args_dict:
            continue
        if not hasattr(getattr(obj, method_name), "__wrapped__"):
            continue

        # f(shared_state, data_samples) looks like a local computation !

        if ("shared_state" in method_args_dict) and ("datasamples" in method_args_dict):
            local_functions_names[key].append(method_name)

        # f(shared_states) looks like an aggregation !
        elif "shared_states" in method_args_dict:
            assert (
                "datasamples" not in method_args_dict
            ), "This method's signature is not valid"
            aggregations_names[key].append(method_name)
        else:
            if method_name not in ("save_local_state", "load_local_state"):
                raise ValueError(
                    "Method {} has a shared_state.s argument but isn't \
                    respecting conventions".format(
                        method_name
                    )
                )

    # Now we have the list of local computations and aggregations names for both
    # strategy and algo.
    # first let's seed the bootstrappping
    if bootstrap_seeds is None:
        bootstrap_seeds_list = np.random.randint(0, 2**32, n_bootstrap)
    else:
        if n_bootstrap is not None:
            assert (
                len(bootstrap_seeds) == n_bootstrap
            ), "bootstrap_seeds must have the same length as n_bootstrap"
        bootstrap_seeds_list = bootstrap_seeds

    # Below is where the magic happens.
    # As a reminder we are trying to hook all caught methods above to make them
    # execute n_bootstrap times on bootstrapped data and then aggregate the
    # n_bootstrap results independently.
    # There are two major difficulties here: first of all we have to tie new
    # methods to the proper object which is the new bootstrapped strategy and
    # algo. To deal with this first issue, we will use the versatility of
    # MethodType.
    # Second in substrafl objects reinstantiate themselves using their class
    # and their kwargs attributes:
    # new_my_object = my_object.__class__(**my_object.kwargs).
    # This is mainly legacy and was done for serialization issues.
    # So we will need to overwrite the original class and not the instance.

    # We have to overwrite the original methods at the class level
    class BtstAlgo(strategy.algo.__class__):
        def __init__(self, *args, bootstrap_specific_kwargs=None, **kwargs):
            super().__init__(*args, **kwargs)
            if bootstrap_specific_kwargs is not None:
                assert len(bootstrap_seeds_list) == len(bootstrap_specific_kwargs), (
                    "bootstrap_specific_kwargs must have the same length as"
                    "bootstrap_seeds_list"
                )
                self.bootstrap_specific_kwargs = bootstrap_specific_kwargs
                self.kwargs.update(
                    "bootstrap_specific_kwargs", bootstrap_specific_kwargs
                )

            self.seeds = bootstrap_seeds_list
            self.individual_algos = []
            for idx in range(len(self.seeds)):
                current_kwargs = copy.deepcopy(strategy.algo.kwargs)
                if self.bootstrap_specific_kwargs is not None:
                    current_kwargs.update(**self.bootstrap_specific_kwargs[idx])
                self.individual_algos.append(
                    copy.deepcopy(strategy.algo.__class__(**current_kwargs))
                )
            # Now we have to overwrite the original methods
            # to be calling their local version on each individual algo
            for local_name in local_functions_names["algo"]:
                f = types.MethodType(
                    _bootstrap_local_function(
                        local_name, bootstrap_function=bootstrap_function
                    ),
                    self,
                )
                setattr(self, local_name, f)

            for agg_name in aggregations_names["algo"]:
                f = types.MethodType(_aggregate_all_bootstraps(agg_name), self)
                setattr(self, agg_name, f)

        def save_local_state(self, path: Path) -> "TorchAlgo":
            # We save all bootstrapped states in different subfolders
            # It assumes at this point checkpoints_list has been populated
            # if it exists

            # The reason for the if is because of initialize functions which don't
            # populate checkpoints_list
            with tempfile.TemporaryDirectory() as tmpdirname:
                paths_to_checkpoints = []
                for idx, algo in enumerate(self.individual_algos):
                    path_to_checkpoint = Path(tmpdirname) / f"bootstrap_{idx}"
                    algo.save_local_state(path_to_checkpoint)
                    paths_to_checkpoints.append(path_to_checkpoint)

                with zipfile.ZipFile(path, "w") as f:
                    for chkpt in paths_to_checkpoints:
                        f.write(chkpt, compress_type=zipfile.ZIP_DEFLATED)
            return self

        def load_local_state(self, path: Path) -> "TorchAlgo":
            """Load the stateful arguments of this class.

            Child classes do not need to
            override that function.

            Parameters
            ----------
                path : pathlib.Path
                    The path where the class has been saved.

            Returns
            -------
                TorchAlgo
                    The class with the loaded elements.
            """
            # Note that at the end of this loop the main state is the one of the last
            # bootstrap
            archive = zipfile.ZipFile(path, "r")
            with tempfile.TemporaryDirectory() as tmpdirname:
                archive.extractall(tmpdirname)
                checkpoints_found = sorted(
                    [p for p in Path(tmpdirname).glob("**/bootstrap_*")]
                )
                self.checkpoints_list = [None] * len(checkpoints_found)
                for idx, file in enumerate(checkpoints_found):
                    self.individual_algos[idx].load_local_state(file)

            return self

        @remote_data
        def predict(
            self, datasamples, shared_state=None, predictions_path: os.PathLike = None
        ):
            predictions = []
            with tempfile.TemporaryDirectory() as tmpdirname:
                paths_to_preds = []
                for idx, algo in enumerate(self.individual_algos):
                    path_to_pred = Path(tmpdirname) / f"bootstrap_{idx}"
                    algo.predict(
                        datasamples=datasamples,
                        shared_state=shared_state,
                        predictions_path=path_to_pred,
                        _skip=True,
                    )
                    paths_to_preds.append(path_to_pred)
                with zipfile.ZipFile(predictions_path, "w") as f:
                    for pred in paths_to_preds:
                        f.write(pred, compress_type=zipfile.ZIP_DEFLATED)

            return predictions

    btst_algo = BtstAlgo(**strategy.algo.kwargs)

    class BtstStrategy(strategy.__class__):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seeds = bootstrap_seeds_list
            self.individual_strategies = []
            for _ in self.seeds:
                # We have to make sure they are independent and new
                self.individual_strategies.append(copy.deepcopy(strategy))
            for local_name in local_functions_names["strategy"]:
                f = types.MethodType(
                    _bootstrap_local_function(
                        local_name,
                        task_type="strategy",
                        bootstrap_function=bootstrap_function,
                    ),
                    self,
                )
                setattr(self, local_name, f)
            for agg_name in aggregations_names["strategy"]:
                f = types.MethodType(
                    _aggregate_all_bootstraps(agg_name, task_type="strategy"),
                    self,
                )
                setattr(self, agg_name, f)

    strategy_kwargs_wo_algo = copy.deepcopy(strategy.kwargs)
    strategy_kwargs_wo_algo.pop("algo")
    return BtstStrategy(algo=btst_algo, **strategy_kwargs_wo_algo), bootstrap_seeds_list


def _bootstrap_local_function(
    local_name: str,
    task_type: str = "algo",
    bootstrap_function: Union[None, Callable] = None,
):
    """Bootstrap the local functiion given.

    Create a new function that bootstrap the given local function given as parameter.
    The idea is to create a method decorated by @remote.

    Parameters
    ----------
    local_name : str
        The atomic task name to be bootstrapped.

    task_type : str
        The type of task to be bootstrapped, either 'algo' or 'strategy'.

    bootstrap_function : Union[None, Callable]
        A function with signature f(datasamples, seed) that returns a bootstrapped
        version of the data.
        If None, use the BootstrapMixin function.
        Note that this can be used to provide splits/cross-validation capabilities
        as well where seed would be the fold number in a flattened list of folds.

    Returns
    -------
    Callable
        The @remote_data function, that has been renamed,
        and will be used as method.
    """
    assert task_type in set(
        ["algo", "strategy"]
    ), "task_type must be either 'algo' or 'strategy'"
    individual_task_type = (
        "individual_algos" if task_type == "algo" else "individual_strategies"
    )
    if bootstrap_function is None:
        # TODO make it cleaner by making bootstrap_sample a function in utils and
        # not a method of BootstrapMixin
        bootstrap_function = partial(
            BootstrapMixin.bootstrap_sample.__func__, self=None
        )

    def local_computation(self, datasamples, shared_state=None) -> list:
        """Execute all the parallel local computations of merged strategies.

        This method is transformed by the decorator to meet Substra API,
        and is executed in the training nodes. See build_graph.

        Parameters
        ----------
        self : MergedStrategy
            The mergedStrategy instance.
        datasamples : pd.DataFrame
            Dataframe returned by the opener.
        shared_state : None, optional
            Given by the aggregation node, so here it is a list of
            the shared_states that have been returned by the individual
            aggregation functions ran at previous step.
            None for the first step, by default None.

        Returns
        -------
        dict
            Local results to be shared via shared_state to the aggregation node.
        """
        results = []

        for idx, seed in enumerate(self.seeds):
            bootstrapped_data = bootstrap_function(datasamples, seed=seed)
            if shared_state is None:
                res = getattr(getattr(self, individual_task_type)[idx], local_name)(
                    datasamples=bootstrapped_data, _skip=True
                )
            else:
                res = getattr(getattr(self, individual_task_type)[idx], local_name)(
                    datasamples=bootstrapped_data,
                    shared_state=shared_state[idx],
                    _skip=True,
                )
            results.append(res)
        return results

    # We need to change its name before decorating it,
    # as substrafl use this name to call the method via getattr afterward.
    local_computation.__name__ = local_name

    return remote_data(local_computation)


def _aggregate_all_bootstraps(aggregation_function_name, task_type: str = "algo"):
    """Aggregate results of bootstraps.

    Create a new function that aggregates each element of a list independently
    using the provided aggregation function and then return the list of results.

    Parameters
    ----------
    aggregation_function_name : str
        The aggregation function to use.

    task_type : str
        The type of task to be bootstrapped, either 'algo' or 'strategy'.

    Returns
    -------
    Callable
        The @remote function, that has been renamed,
        and will be used as method.
    """
    assert task_type in set(
        ["algo", "strategy"]
    ), "task_type must be either 'algo' or 'strategy'"
    individual_task_type = (
        "individual_algos" if task_type == "algo" else "individual_strategies"
    )

    def aggregation(self, shared_states=None) -> list:
        """Execute all the parallel aggregations of independent bootstrap runs.

        Parameters
        ----------
        self : MergedStrategy
            The mergedStrategy instance.

        shared_states : List
            List of lists of results returned by local_computation ran at
            previous step.
            The first axis is the local_computations ran by the different
            strategies merged,
            and the second one is for partners.results  from training nodes.
            So shared_states[0][1] corresponds to the output of the
            local_computation of the first (idx 0) merged strategy,
            computed on the second (idx 1) client.

        Returns
        -------
        dict
            Global results to be shared with train nodes via shared_state.
        """
        results = []
        if shared_states is not None:
            # loop over the aggregation steps provided using _skip=True
            for idx, shared_state in enumerate(zip(*shared_states)):
                res = getattr(
                    getattr(self, individual_task_type)[idx], aggregation_function_name
                )(shared_states=shared_state, _skip=True)
                results.append(res)
        else:
            # This is the case in initialize
            results = []
            for task in getattr(self, individual_task_type):
                res = getattr(task, aggregation_function_name)(
                    shared_states=None, _skip=True
                )
                results.append(res)

            if all([res is None for res in results]):
                results = None

        return results

    aggregation.__name__ = aggregation_function_name
    return remote(aggregation)


def make_bootstrap_metric_function(metric_function):
    """Averages metric on each bootstrapped versions of the models.

    Parameters
    ----------
    metric_function : list
        The metric function to hook.
    """

    def bootstraped_metric(datasamples, predictions_path):
        list_of_metrics = []
        if isinstance(predictions_path, str) or isinstance(predictions_path, Path):
            archive = zipfile.ZipFile(predictions_path, "r")
            with tempfile.TemporaryDirectory() as tmpdirname:
                archive.extractall(tmpdirname)
                preds_found = sorted(
                    [p for p in Path(tmpdirname).glob("**/bootstrap_*")]
                )
                for pred_found in preds_found:
                    list_of_metrics.append(metric_function(datasamples, pred_found))
        else:
            y_preds = predictions_path
            for y_pred in y_preds:
                list_of_metrics.append(metric_function(datasamples, y_pred))
        return np.array(list_of_metrics).mean()

    return bootstraped_metric


if __name__ == "__main__":
    from substrafl.algorithms.pytorch import TorchFedAvgAlgo  # , TorchNewtonRaphsonAlgo
    from substrafl.dependency import Dependency
    from substrafl.evaluation_strategy import EvaluationStrategy
    from substrafl.experiment import execute_experiment
    from substrafl.index_generator import NpIndexGenerator
    from substrafl.nodes import AggregationNode
    from substrafl.strategies import FedAvg  # , NewtonRaphson

    from fedeca import LogisticRegressionTorch
    from fedeca.utils import make_accuracy_function, make_substrafl_torch_dataset_class
    from fedeca.utils.data_utils import split_dataframe_across_clients
    from fedeca.utils.survival_utils import CoxData

    seed = 42
    torch.manual_seed(seed)
    # Number of model updates between each FL strategy aggregation.
    NUM_UPDATES = 100

    # Number of samples per update.
    BATCH_SIZE = 32

    index_generator = NpIndexGenerator(
        batch_size=BATCH_SIZE,
        num_updates=NUM_UPDATES,
    )

    # Let's generate 1000 data samples with 10 covariates
    data = CoxData(seed=42, n_samples=1000, ndim=50, overlap=10.0, propensity="linear")
    df = data.generate_dataframe()

    # We remove the true propensity score
    df = df.drop(columns=["propensity_scores"], axis=1)

    class UnifLogReg(LogisticRegressionTorch):
        """Spawns FedECA logreg model with uniform weights."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fc1.weight.data.uniform_(-1, 1)

    logreg_model = UnifLogReg(ndim=50)
    optimizer = torch.optim.Adam(logreg_model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    logreg_dataset_class = make_substrafl_torch_dataset_class(
        ["treatment"],
        event_col="event",
        duration_col="time",
        return_torch_tensors=True,
    )
    accuracy = make_accuracy_function("treatment")
    accuracy_btst = make_bootstrap_metric_function(accuracy)

    class TorchLogReg(TorchFedAvgAlgo):
        """Spawns FedAvg algo with logreg model with uniform weights."""

        def __init__(self):
            super().__init__(
                model=logreg_model,
                criterion=criterion,
                optimizer=optimizer,
                index_generator=index_generator,
                dataset=logreg_dataset_class,
                seed=seed,
                use_gpu=False,
            )

    # class TorchLogReg(TorchNewtonRaphsonAlgo):
    #     def __init__(self):
    #         super().__init__(
    #             model=logreg_model,
    #             criterion=criterion,
    #             batch_size=32,
    #             dataset=logreg_dataset_class,
    #             seed=seed,
    #             use_gpu=False,
    #         )

    strategy = FedAvg(algo=TorchLogReg())

    btst_strategy, _ = make_bootstrap_strategy(strategy, n_bootstrap=10)

    clients, train_data_nodes, test_data_nodes, _, _ = split_dataframe_across_clients(
        df,
        n_clients=2,
        split_method="uniform",
        split_method_kwargs=None,
        data_path="./data",
        backend_type="subprocess",
    )

    for node in test_data_nodes:
        node.metric_functions = {accuracy_btst.__name__: accuracy_btst}

    first_key = list(clients.keys())[0]

    aggregation_node = AggregationNode(
        clients[first_key].organization_info().organization_id
    )

    dependencies = Dependency(
        pypi_dependencies=[
            "numpy==1.24.3",
            "scikit-learn==1.3.1",
            "torch==2.0.1",
            "--extra-index-url https://download.pytorch.org/whl/cpu",
        ]
    )
    # Test at the end of every round
    my_eval_strategy = EvaluationStrategy(
        test_data_nodes=test_data_nodes, eval_frequency=1
    )
    xp_dir = str(Path.cwd() / "tmp" / "experiment_summaries")
    os.makedirs(xp_dir, exist_ok=True)

    compute_plan = execute_experiment(
        client=clients[first_key],
        strategy=btst_strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=my_eval_strategy,
        aggregation_node=aggregation_node,
        num_rounds=10,
        experiment_folder=xp_dir,
        dependencies=dependencies,
        clean_models=False,
        name="FedECA",
    )
    print(pd.DataFrame(clients[first_key].get_performances(compute_plan.key).dict()))
