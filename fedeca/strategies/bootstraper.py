import copy
import inspect
import os
import re
import tempfile
import types
import zipfile
from pathlib import Path
from typing import Union

import numpy as np
import torch
from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.remote import remote, remote_data
from substrafl.strategies.strategy import Strategy


def make_bootstrap_strategy(
    strategy: Strategy,
    n_bootstraps: Union[int, None] = None,
    bootstrap_seeds: Union[list[int], None] = None,
    inplace: bool = False,
):
    """Bootstrap a substrafl strategy wo impacting the number of compute tasks.

    In order to reduce the bottleneck of substra when bootstraping a strategy
    we need to go over the strategy compute plan and modifies each local atomic
    task to execute n_bootstraps times on bootstraped data. Each modified task
    returns a list of n_bootstraps original outputs obtained on each bootstrap.
    Each aggregation task is then modified to aggregate the n_bootstraps outputs
    independently.
    This code heavily uses some code patterns invented by Arthur Pignet.

    Parameters
    ----------
    strategy : Strategy
        The strategy to bootstrap.
    n_bootstraps : Union[int, None]
        Number of bootstrap to be performed. If None will use
        len(bootstrap_seeds) instead. If bootstrap_seeds is given
        seeds those seeds will be used for the generation
        otherwise seeds are generated randomly.
    bootstrap_seeds : Union[list[int], None]
        The list of seeds used for bootstrapping random states.
        If None will generate n_bootstraps randomly, in the presence
        of both allways use bootstrap_seeds.
    inplace : bool, optional
        Whether to modify the strategy inplace or not, by default False.
    Returns
    -------
    Strategy
        The resulting efficiently bootstrapped strategy
    """
    # We dynamically get all methods from strategy except 'magic' methods
    if not inplace:
        strategy = copy.deepcopy(strategy)
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

    # We need to differentiate between aggregations and local computations
    # we'll use the signature of the methods
    # note that there is no way to know the order in which they are applied

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
        method_args_dict = inspect.signature(getattr(obj, method_name)).parameters
        if not (
            ("shared_states" in method_args_dict)
            or ("shared_state" in method_args_dict)
        ) or (method_name in ["save_local_state", "load_local_state"]):
            continue
        if "predictions_path" in method_args_dict:
            continue
        # We create a copy of all methods with the original suffix to avoid name
        # collision and infinite recursion when decorating the old methods

        setattr(obj, method_name + "_original", getattr(obj, method_name))

        if ("shared_state" in method_args_dict) and ("datasamples" in method_args_dict):
            local_functions_names[key].append(method_name)
        elif "shared_states" in method_args_dict:
            aggregations_names[key].append(method_name)
        else:
            if method_name not in ("save_local_state", "load_local_state"):
                raise ValueError(
                    "Method {} has a shared_state.s argument but isn't \
                    respecting conventions".format(
                        method_name
                    )
                )

    # Now we are totally free to modify the original methods inplace
    # We need to differentiate between aggregations and local computations
    # but first let's seed the bootstrappping
    if bootstrap_seeds is None:
        bootstrap_seeds_list = np.random.randint(0, 2**32, n_bootstraps)
    else:
        if n_bootstraps is not None:
            assert (
                len(bootstrap_seeds) == n_bootstraps
            ), "bootstrap_seeds must have the same length as n_bootstraps"
        bootstrap_seeds_list = bootstrap_seeds

    orig_local_fcts = {}
    orig_aggregations = {}
    for key in ["strategy", "algo"]:
        if key == "strategy":
            obj = strategy
        else:
            obj = strategy.algo
        orig_local_fcts[key] = [
            getattr(obj, name) for name in local_functions_names[key]
        ]
        orig_aggregations[key] = [
            getattr(obj, name) for name in aggregations_names[key]
        ]

    local_computations_fct = {}
    aggregations_fct = {}
    # Define the merged methods using the functions defined above.
    for key in ["strategy", "algo"]:
        # most important line as substrafl will never use instances themselves
        local_computations_fct[key] = [
            _bootstrap_local_function(
                local_function, name, bootstrap_seeds_list=bootstrap_seeds_list
            )
            for local_function, name in zip(
                orig_local_fcts[key], local_functions_names[key]
            )
        ]
        aggregations_fct[key] = [
            _aggregate_all_bootstraps(agg, name)
            for agg, name in zip(orig_aggregations[key], aggregations_names[key])
        ]

    # We have to overwrite the original methods at the class level
    class BtstAlgo(strategy.algo.__class__):
        def __init__(self, *args, **kwargs):

            super().__init__(*args, **kwargs)
            for local_name in local_functions_names["algo"]:
                setattr(self, local_name + "_original", getattr(self, local_name))
                f = types.MethodType(
                    _bootstrap_local_function(
                        getattr(self, local_name), local_name, bootstrap_seeds_list
                    ),
                    self,
                )
                setattr(self, local_name, f)

            for agg_name in aggregations_names["algo"]:
                setattr(self, agg_name + "_original", getattr(self, agg_name))
                f = types.MethodType(
                    _aggregate_all_bootstraps(getattr(self, agg_name), agg_name), self
                )
                setattr(self, agg_name, f)

        def save_local_state(self, path: Path) -> "TorchAlgo":
            # We save all bootstrapped states in different subfolders
            # It assumes at this point checkpoints_list has been populated
            # if it exists

            # The reason for the if is because of initialize functions which don't
            # populate checkpoints_list
            if hasattr(self, "checkpoints_list"):
                pass
            else:
                self.checkpoints_list = [
                    copy.deepcopy(self._get_state_to_save())
                    for _ in range(len(bootstrap_seeds_list))
                ]
            with tempfile.TemporaryDirectory() as tmpdirname:
                paths_to_checkpoints = []
                for idx, checkpt in enumerate(self.checkpoints_list):
                    # Get the model in the proper state
                    self._update_from_checkpoint(checkpt)
                    assert not checkpt
                    path_to_checkpoint = Path(tmpdirname) / f"bootstrap_{idx}"
                    super(strategy.algo.__class__, self).save_local_state(
                        path_to_checkpoint
                    )
                    paths_to_checkpoints.append(path_to_checkpoint)

                with zipfile.ZipFile(path, "w") as f:
                    for chkpt in paths_to_checkpoints:
                        f.write(chkpt, compress_type=zipfile.ZIP_DEFLATED)
            return self

        def load_local_state(self, path: Path) -> "TorchAlgo":
            """Load the stateful arguments of this class. Child classes do not need to
            override that function.

            Args:
                path (pathlib.Path): The path where the class has been saved.

            Returns:
                TorchAlgo: The class with the loaded elements.
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
                    super(strategy.algo.__class__, self).load_local_state(file)
                    self.checkpoints_list[idx] = copy.deepcopy(
                        self._get_state_to_save()
                    )
            return self

        @remote_data
        def predict(
            self, datasamples, shared_state=None, predictions_path: os.PathLike = None
        ):
            predictions = []
            with tempfile.TemporaryDirectory() as tmpdirname:
                paths_to_preds = []
                for idx, chckpt in enumerate(self.checkpoints_list):
                    self._update_from_checkpoint(chckpt)
                    path_to_pred = Path(tmpdirname) / f"bootstrap_{idx}"
                    super(strategy.algo.__class__, self).predict(
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
            for local_name in local_functions_names["strategy"]:
                setattr(self, local_name + "_original", getattr(self, local_name))
                f = types.MethodType(
                    _bootstrap_local_function(
                        getattr(self, local_name), local_name, bootstrap_seeds_list
                    ),
                    self,
                )
                setattr(self, local_name, f)
            for agg_name in aggregations_names["strategy"]:
                setattr(self, agg_name + "_original", getattr(self, agg_name))
                f = types.MethodType(
                    _aggregate_all_bootstraps(getattr(self, agg_name), agg_name), self
                )
                setattr(self, agg_name, f)

    strategy.kwargs.pop("algo")
    return BtstStrategy(algo=btst_algo, **strategy.kwargs), bootstrap_seeds_list


def _bootstrap_predict(predict):
    def new_predict(self, predictions_path):
        return self

    return new_predict


def _bootstrap_local_function(local_function, new_op_name, bootstrap_seeds_list):
    """Bootstrap the local functiion given.

    Create a new function that bootstrap the given local function given as parameter.
    The idea is to create a method decorated by @remote.

    Parameters
    ----------
    local_function : function
        The atomic task to be bootstrapped.
    new_op_name : str
        name of the method.
    bootstrap_seeds_list : list[int]
        The list of seeds used for bootstrapping.

    Returns
    -------
    function
        The @remote_data function, that has been renamed,
        and will be used as method.
    """

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
        # loop over the provided local_computation steps using skip=True.
        # What is highly non-trivial is that algo has a state that is bootstrap
        # dependent and we need to load the corresponding state as the main
        # state, so we need to have saved all states (aka i.e. n_bootstraps models)
        # We use implicitly the new method load_bootstrap_states to load all
        # states in-RAM
        name_decorated_function = local_computation.__name__ + "_original"
        if not hasattr(self, "checkpoints_list"):
            self.checkpoints_list = [None] * len(bootstrap_seeds_list)

        for idx, seed in enumerate(bootstrap_seeds_list):
            rng = np.random.default_rng(seed)
            bootstrapped_data = datasamples.sample(
                datasamples.shape[0], replace=True, random_state=rng
            )
            from remote_pdb import RemotePdb

            RemotePdb("127.0.0.1", 4444).set_trace()
            # Loading the correct state into the current main algo
            if self.checkpoints_list[idx] is not None:
                self._update_from_checkpoint(self.checkpoints_list[idx])
            # We need this old state to avoid side effects from the function
            # on the instance
            old_state = copy.deepcopy(self)
            if shared_state is None:
                res = getattr(self, name_decorated_function)(
                    datasamples=bootstrapped_data, _skip=True
                )
            else:
                res = getattr(self, name_decorated_function)(
                    datasamples=bootstrapped_data,
                    shared_state=shared_state[idx],
                    _skip=True,
                )

            self.checkpoints_list[idx] = copy.deepcopy(self._get_state_to_save())

            # We restore the algo to its old state
            for att_name, att in vars(self).items():
                if att_name == "checkpoints_list":
                    continue

                def equality_check(a, b):
                    if type(a) != type(b):
                        return False
                    else:
                        if isinstance(a, dict):
                            for k in a.keys():
                                if not equality_check(a[k], b[k]):
                                    return False
                            return True
                        elif isinstance(a, list):
                            for i in range(len(a)):
                                if not equality_check(a[i], b[i]):
                                    return False
                            return True
                        elif isinstance(a, np.ndarray):
                            return np.all(a == b)
                        elif isinstance(a, torch.Tensor):
                            return torch.all(a == b)
                        else:
                            return a == b

                if not equality_check(att, old_state.__getattribute__(att_name)):
                    self.__setattr__(att_name, old_state.__getattribute__(att_name))

            results.append(res)
        return results

    # We need to change its name before decorating it,
    # as substrafl use this name to call the method via getattr afterward.
    local_computation.__name__ = new_op_name

    return remote_data(local_computation)


def _aggregate_all_bootstraps(aggregation_function, new_op_name):
    """Aggregate results of bootstraps.

    Create a new function that aggregates each element of a list independently
    using the provided aggregation function and then return the list of results.

    Parameters
    ----------
    aggregation_function : function
        The aggregation function to use.
    new_op_name : str
        name of the method.

    Returns
    -------
    function
        The @remote function, that has been renamed,
        and will be used as method.
    """

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
        name_decorated_function = aggregation_function.__name__ + "_original"
        if shared_states is not None:
            # loop over the aggregation steps provided using _skip=True
            for shared_state in zip(*shared_states):
                res = getattr(self, name_decorated_function)(
                    shared_states=shared_state, _skip=True
                )
                results.append(res)
        else:
            # This is the case in initialize
            results = getattr(self, name_decorated_function)(
                shared_states=None, _skip=True
            )

        return results

    aggregation.__name__ = new_op_name
    return remote(aggregation)


def make_bootstrap_metric_function(metric_function):
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
    import pandas as pd
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

    btst_strategy, _ = make_bootstrap_strategy(strategy, n_bootstraps=10)

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
