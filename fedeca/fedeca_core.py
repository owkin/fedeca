"""Federate causal inference on distributed data."""
import copy
import logging
import sys
import time
from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from scipy.linalg import inv
from substra.sdk.models import ComputePlanStatus
from substrafl.algorithms.pytorch import TorchNewtonRaphsonAlgo
from substrafl.model_loading import download_algo_state
from substrafl.nodes import AggregationNode, TrainDataNode
from substrafl.strategies import FedAvg, NewtonRaphson
from torch import nn
from torch.optim import SGD, Optimizer

from fedeca.algorithms import TorchWebDiscoAlgo
from fedeca.algorithms.torch_dp_fed_avg_algo import TorchDPFedAvgAlgo
from fedeca.analytics import RobustCoxVariance, RobustCoxVarianceAlgo
from fedeca.strategies import WebDisco
from fedeca.strategies.bootstraper import make_bootstrap_strategy
from fedeca.strategies.webdisco_utils import (
    compute_summary_function,
    get_final_cox_model_function,
)
from fedeca.utils import (
    Experiment,
    make_accuracy_function,
    make_c_index_function,
    make_substrafl_torch_dataset_class,
)
from fedeca.utils.bootstrap_utils import make_bootstrap_function
from fedeca.utils.data_utils import split_dataframe_across_clients
from fedeca.utils.substra_utils import Client
from fedeca.utils.substrafl_utils import (
    get_outmodel_function,
    get_simu_state_from_round,
)
from fedeca.utils.survival_utils import (
    BaseSurvivalEstimator,
    BootstrapMixin,
    CoxPHModelTorch,
)

logger = logging.getLogger(__name__)


class FedECA(Experiment, BaseSurvivalEstimator, BootstrapMixin):
    """FedECA class tthat performs Federated IPTW."""

    def __init__(
        self,
        ndim: int,
        ds_client=None,
        train_data_nodes: Union[list[TrainDataNode], None] = None,
        treated_col: str = "treated",
        event_col: str = "E",
        duration_col: str = "T",
        ps_col="propensity_scores",
        num_rounds_list: list[int] = [10, 10],
        damping_factor_nr: float = 0.8,
        l2_coeff_nr: float = 0.0,
        standardize_data: bool = True,
        penalizer: float = 0.0,
        l1_ratio: float = 1.0,
        initial_step_size: float = 0.95,
        learning_rate_strategy: str = "lifelines",
        dtype: float = "float64",
        propensity_strategy="iptw",
        variance_method: str = "naïve",
        n_bootstrap: Union[int, None] = 200,
        bootstrap_seeds: Union[list[int], None] = None,
        bootstrap_function: Union[Callable, str] = "global",
        clients_sizes: Union[list, None] = None,
        client_identifier: str = "client",
        clients_names: Union[list, None] = None,
        dp_target_epsilon: Union[float, None] = None,
        dp_target_delta: Union[float, None] = None,
        dp_max_grad_norm: Union[float, None] = None,
        dp_propensity_model_optimizer_class: Optimizer = SGD,
        dp_propensity_model_optimizer_kwargs: Union[dict, None] = None,
        dp_propensity_model_training_params: Union[dict, None] = None,
        seed: int = 42,
        aggregation_node: Union[AggregationNode, None] = None,
        experiment_folder: str = "./iptw_experiment",
        clean_models: bool = False,
        dependencies: Union[list, None] = None,
        timeout: int = 3600,
        sleep_time: int = 30,
        fedeca_path: Union[None, str] = None,
        evaluation_frequency=None,
        partner_client: Union[None, Client] = None,
    ):
        """Initialize the Federated IPTW class.

        Implements the FedECA algorithm which combines
        an estimation of propensity scores using logistic regression
        and the fit of a weighted Cox Model in a federated fashion.

        Parameters
        ----------
        client : fl.client.Client
            Federated Learning client object.
        train_data_nodes : list
            List of data nodes participating in the federated training.
        ndim : int
            Number of dimensions (features) in the dataset.
        treated_col : str, optional
            Column name indicating treatment status, by default "treated".
        event_col : str, optional
            Column name indicating event occurrence, by default "E".
        duration_col : str, optional
            Column name indicating time to event or censoring, by default "T".
        num_rounds_list : list, optional
            List of number of rounds for each stage, by default [10, 10].
        damping_factor_nr : float, optional
            Damping factor for natural gradient regularization, by default 0.8.
        l2_coeff_nr : float, optional
            L2 regularization coefficient for natural gradient, by default 0.0.
        standardize_data : bool, optional
            Whether to standardize data before training, by default True.
        penalizer : float, optional
            Penalizer for IPTW objective, by default 0.0.
        l1_ratio : float, optional
            L1 ratio for IPTW objective, by default 1.0.
        initial_step_size : float, optional
            Initial step size for optimization, by default 0.95.
        learning_rate_strategy : str, optional
            Learning rate strategy, by default "lifelines".
        batch_size : int, optional
            Batch size for optimization, by default sys.maxsize.
        dtype : str, optional
            Data type for the model, by default "float64".
        propensity_strategy: str, optional
            The propensity strategy to use.
        variance_method : `{"naive", "robust", "bootstrap"}`
            Method for estimating the variance, and therefore the p-value of the
            estimated treatment effect.
            * "naive": Inverse of the Fisher information.
            * "robust": The robust sandwich estimator [1] computed in FL thanks
                to FedECA. Useful when samples are reweighted.
            * "bootstrap": Bootstrap the given data by sampling each patient
              with replacement, each time estimate the treatment effect, then
              use all repeated estimations to compute the variance. The implementation
              is efficient in substra and shouldn't induce too much overhead.
            Defauts to naïve.
            [1] David A Binder. Fitting cox’s proportional hazards models from survey data. Biometrika, 79(1):139–147, 1992.  # noqa: E501
        n_bootstrap : Union[int, None]
            Number of bootstrap to be performed. If None will use
            len(bootstrap_seeds) instead. If bootstrap_seeds is given
            seeds those seeds will be used for the generation
            otherwise seeds are generated randomly.
        bootstrap_seeds : Union[list[int], None]
            The list of seeds used for bootstrapping random states.
            If None will generate n_bootstrap randomly, in the presence
            of both allways use bootstrap_seeds.
        bootstrap_function : Union[Callable, str]
            The bootstrap function to use for instance if it is necessary to
            mimic perfectly a global sampling. For now two are implemented by
            default "global" and "per-client", which samples with replacement
            from the global data or from the client data respectively using
            the seeds given in bootstrap_seeds.
        clients_sizes : Union[list, None]
            The sizes of the clients to use for the global bootstrap need to be
            ordered exactly as the train data nodes.
        client_identifier: str
            The column name which contains the client identifier for the global
            bootstrap. By default "client".
        clients_names: Union[list, None]
            The different names found in the client_identifier column to categorize
            each client.
        dp_target_epsilon: float
            The target epsilon for (epsilon, delta)-differential
            private guarantee. Defaults to None.
        dp_target_delta: float
            The target delta for (epsilon, delta)-differential
            private guarantee. Defaults to None.
        dp_max_grad_norm: float
            The maximum L2 norm of per-sample gradients;
            used to enforce differential privacy. Defaults to None.
        dp_propensity_model_optimizer_class: torch.optim.Optimizer
            The optimizer to use for the training of the propensity model.
            Defauts to Adam.
        dp_propensity_model_optimizer_class_kwargs: dict
            The params to give to optimizer class.
        dp_propensity_model_training_params: dict
            A dict with keys batch_size and num_updates for the DP-SGD training.
            Defaults to None.
        seed : int, optional
            Seed for random number generation, by default 42.
        aggregation_node : str or None, optional
            Node for aggregation, by default None.
        experiment_folder : str, optional
            Folder path for experiment outputs, by default "./iptw_experiment".
        clean_models : bool, optional
            Whether to clean models after training, by default False.
        dependencies : list, optional
            List of dependencies, by default None.
        timeout : int, optional
            Timeout for a single round of federated learning, by default 3600.
        sleep_time : int, optional
            Sleep time between rounds, by default 30.
        fedeca_path:
            Path towards the fedeca reository.
        evaluation_frequency:
            Evaluation_frequency.
        partner_client: Union[None, Client]
            The client of the partner if we are in remote mode.
        """
        self.standardize_data = standardize_data
        assert dtype in ["float64", "float32", "float16"]
        if dtype == "float64":
            self.torch_dtype = torch.float64
        elif dtype == "float32":
            self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch.float16

        self.ndim = ndim
        self.treated_col = treated_col
        self.event_col = event_col
        self.duration_col = duration_col
        self.ps_col = ps_col
        self.seed = seed
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.initial_step_size = initial_step_size
        self.learning_rate_strategy = learning_rate_strategy
        # Careful about mutable default args
        self.num_rounds_list = copy.deepcopy(num_rounds_list)
        self.timeout = timeout
        self.sleep_time = sleep_time
        self.damping_factor_nr = damping_factor_nr
        self.l2_coeff_nr = l2_coeff_nr
        self.propensity_strategy = propensity_strategy
        self.variance_method = variance_method
        self.robust = variance_method == "robust"
        self.is_bootstrap = variance_method == "bootstrap"
        self.n_bootstrap = n_bootstrap
        self.bootstrap_seeds = bootstrap_seeds
        self.clients_sizes = clients_sizes
        self.client_identifier = client_identifier
        self.clients_names = clients_names
        self.bootstrap_function = bootstrap_function

        if isinstance(bootstrap_function, str):
            assert self.bootstrap_function in ["global", "per-client"], "bootstrap"
            f" {self.bootstrap_function} not implemented. Only 'global' and"
            " 'per-client' are possible. If you wish to do custom bootstrapping "
            " use a Callable f(data: pd.DataFrame, seed: int) -> pd.DataFrame."
            assert (
                self.clients_sizes is not None
            ), "You need to provide the size of each client in a list"
            self.clients_names = (
                self.clients_names
                if self.clients_names is not None
                else [f"client{i}" for i in range(len(self.clients_sizes))]
            )
            "of clients for global bootstrap by passing num_clients"
            if bootstrap_function == "global":
                assert self.total_size is not None, "You need to provide the total"
                " size of the dataset for global bootstrap by setting total_size"
                print(
                    "WARNING: global bootstrap necessitates that the opener returns"
                    f" a column {self.client_identifier} with levels="
                    f" {self.clients_names} so that tasks know in which client"
                    " they occur so that they can select the right global"
                    " indices (note that you can set the levels of the column by"
                    " passing clients_names)"
                )
                print("Client identifier column is:", self.client_identifier)
                print("Clients' names found in this column are:", self.clients_names)
                (
                    self.bootstrap_function,
                    self.bootstrap_seeds,
                    _,
                ) = make_global_bootstrap_function(
                    clients_sizes=self.clients_sizes,
                    n_bootstrap=self.n_bootstrap,
                    bootstrap_seeds=self.bootstrap_seeds,
                    client_identifier=self.client_identifier,
                )
            elif bootstrap_function == "per-client":
                self.bootstrap_function = None
        else:
            pass

        self.dp_target_delta = dp_target_delta
        self.dp_target_epsilon = dp_target_epsilon
        self.dp_max_grad_norm = dp_max_grad_norm
        self.dp_propensity_model_training_params = dp_propensity_model_training_params
        self.dp_propensity_model_optimizer_class = dp_propensity_model_optimizer_class
        self.dp_propensity_model_optimizer_kwargs = dp_propensity_model_optimizer_kwargs
        self.dependencies = dependencies
        self.experiment_folder = experiment_folder
        self.fedeca_path = fedeca_path
        self.evaluation_frequency = evaluation_frequency
        self.partner_client = partner_client
        self.dtype = dtype

        kwargs = {}
        kwargs["algo_dependencies"] = self.dependencies
        self.accuracy_metrics_dict = {
            "accuracy": make_accuracy_function(self.treated_col)
        }
        self.cindex_metrics_dict = {
            "C-index": make_c_index_function(
                event_col=self.event_col, duration_col=self.duration_col
            )
        }

        # Note that we don't use self attributes because substrafl classes are messed up
        # and we don't want confusion
        self.logreg_model = LogisticRegressionTorch(self.ndim, self.torch_dtype)
        self.logreg_dataset_class = make_substrafl_torch_dataset_class(
            [self.treated_col],
            self.event_col,
            self.duration_col,
            dtype=dtype,
            return_torch_tensors=True,
        )
        # Set propensity model training to DP or not DP mode
        self.set_propensity_model_strategy()

        # We use only the treatment variable in the model
        cox_model = CoxPHModelTorch(ndim=1, torch_dtype=self.torch_dtype)
        survival_dataset_class = make_substrafl_torch_dataset_class(
            [self.duration_col, self.event_col],
            self.event_col,
            self.duration_col,
            dtype=dtype,
        )

        # no self attributes in this class !!!!!!
        class WDAlgo(TorchWebDiscoAlgo):
            def __init__(self, propensity_model, robust):
                super().__init__(
                    model=cox_model,
                    # TODO make this batch-size argument disappear from
                    # webdisco algo
                    batch_size=sys.maxsize,
                    dataset=survival_dataset_class,
                    seed=seed,
                    duration_col=duration_col,
                    event_col=event_col,
                    treated_col=treated_col,
                    standardize_data=standardize_data,
                    penalizer=penalizer,
                    l1_ratio=l1_ratio,
                    initial_step_size=initial_step_size,
                    learning_rate_strategy=learning_rate_strategy,
                    store_hessian=True,
                    propensity_model=propensity_model,
                    propensity_strategy=propensity_strategy,
                    robust=robust,
                )
                self._propensity_model = propensity_model

        self.webdisco_algo = WDAlgo(propensity_model=None, robust=self.robust)
        self.webdisco_strategy = WebDisco(
            algo=self.webdisco_algo,
            standardize_data=self.standardize_data,
            metric_functions=self.cindex_metrics_dict,
        )
        strategies_to_run = [self.propensity_model_strategy, self.webdisco_strategy]

        if self.variance_method == "bootstrap":
            # !!!! WARNING !!!!!!
            # We only bootstrap the first strategy because
            # for the glue to work for the second one we need to
            # do it later and there is no 3rd one with bootstrap
            strategies_to_run = [
                make_bootstrap_strategy(
                    strategies_to_run[0],
                    n_bootstrap=self.n_bootstrap,
                    bootstrap_seeds=self.bootstrap_seeds,
                    bootstrap_function=self.bootstrap_function,
                )[0]
            ] + strategies_to_run[1:]

        kwargs["strategies"] = strategies_to_run
        if self.robust:
            # We prepare robust estimation
            class MockAlgo:
                def __init__(self):
                    self.strategies = ["Robust Cox Variance"]

            mock_algo = MockAlgo()
            kwargs["strategies"].append(
                RobustCoxVariance(algo=mock_algo, metric_functions={})
            )
            # We need this line for the zip to consider all 3
            # strategies
            self.num_rounds_list.append(sys.maxsize)

        kwargs["ds_client"] = ds_client
        kwargs["train_data_nodes"] = train_data_nodes
        kwargs["aggregation_node"] = aggregation_node
        kwargs["experiment_folder"] = self.experiment_folder
        kwargs["clean_models"] = clean_models
        kwargs["num_rounds_list"] = self.num_rounds_list
        kwargs["fedeca_path"] = self.fedeca_path
        kwargs["algo_dependencies"] = self.dependencies
        kwargs["evaluation_frequency"] = self.evaluation_frequency
        kwargs["partner_client"] = self.partner_client

        # TODO: test_data_nodes and evaluation_frequency are not passed

        super().__init__(**kwargs)

    def check_cp_status(self, idx=0):
        """Check the status of the process."""
        training_type = "training"
        if idx == 0:
            model_name = "Propensity Model"
        elif idx == 1:
            model_name = "Weighted Cox Model"
        else:
            model_name = "Robust Variance"
            training_type = "estimation"

        logger.info(f"Waiting on {model_name} {training_type} to finish...")
        t1 = time.time()
        t2 = t1
        while (t2 - t1) < self.timeout:
            status = self.ds_client.get_compute_plan(
                self.compute_plan_keys[idx].key
            ).status
            if status == ComputePlanStatus.done:
                logger.info(
                    f"""Compute plan {self.compute_plan_keys[0].key} of {model_name} has
                    finished !"""
                )
                break
            elif (
                status == ComputePlanStatus.failed
                or status == ComputePlanStatus.canceled
            ):
                raise ValueError(
                    f"""Compute plan {self.compute_plan_keys[0].key} of {model_name} has
                    failed"""
                )
            elif (
                status == ComputePlanStatus.doing
                or status == ComputePlanStatus.todo
                or status == ComputePlanStatus.waiting
            ):
                pass
            else:
                logger.warning(
                    f"""Compute plan status is {status}, this shouldn't happen, sleeping
                    {self.time_sleep} and retrying until timeout {self.timeout}"""
                )
            time.sleep(self.sleep_time)

    def set_propensity_model_strategy(self):
        """Set FedECA to use DP.

        At the end it sets the parameter self.propensity_model_strateg
        """
        self.dp_params_given = [
            self.dp_max_grad_norm is not None,
            self.dp_target_epsilon is not None,
            self.dp_target_delta is not None,
        ]

        if any(self.dp_params_given) and not all(self.dp_params_given):
            raise ValueError(
                "To use DP you should provide values for all DP parameters: "
                "dp_max_grad_norm, dp_target_epsilon and dp_target_delta"
            )
        self._apply_dp = all(self.dp_params_given)
        if self._apply_dp:
            assert (
                self.dp_propensity_model_training_params is not None
            ), "You should give dp_propensity_model_training_params"
            "={'batch_size': ?, 'num_updates': ?}"
            assert (
                "batch_size" in self.dp_propensity_model_training_params
                and "num_updates" in self.dp_propensity_model_training_params
            ), "You should fill all fields of dp_propensity_model_training_params"
            "={'batch_size': ?, 'num_updates': ?}"
            if self.dp_propensity_model_optimizer_kwargs is None:
                self.dp_propensity_model_optimizer_kwargs = {}
            dp_propensity_model_optimizer = self.dp_propensity_model_optimizer_class(
                params=self.logreg_model.parameters(),
                **self.dp_propensity_model_optimizer_kwargs,
            )
            num_rounds_propensity = self.num_rounds_list[0]

            # no self attributes in this class !!!!!!
            # fed_iptw_self = self hack doesn't work for serialization issue
            logreg_model = self.logreg_model
            logreg_dataset_class = self.logreg_dataset_class
            seed = self.seed
            num_updates = self.dp_propensity_model_training_params["num_updates"]
            batch_size = self.dp_propensity_model_training_params["batch_size"]
            dp_target_epsilon = self.dp_target_epsilon
            dp_target_delta = self.dp_target_delta
            dp_max_grad_norm = self.dp_max_grad_norm

            class DPLogRegAlgo(TorchDPFedAvgAlgo):
                def __init__(self):
                    super().__init__(
                        model=logreg_model,
                        criterion=nn.BCELoss(),
                        optimizer=dp_propensity_model_optimizer,
                        dataset=logreg_dataset_class,
                        seed=seed,
                        num_updates=num_updates,
                        batch_size=batch_size,
                        num_rounds=num_rounds_propensity,
                        dp_target_epsilon=dp_target_epsilon,
                        dp_target_delta=dp_target_delta,
                        dp_max_grad_norm=dp_max_grad_norm,
                    )

            self.dp_algo = DPLogRegAlgo()
            self.dp_strategy = FedAvg(
                algo=self.dp_algo, metric_functions=self.accuracy_metrics_dict
            )
            self.propensity_model_strategy = self.dp_strategy
        else:
            # no self attributes in this class
            # fed_iptw_self = self hack doesn't work for serialization issue
            logreg_model = self.logreg_model
            logreg_dataset_class = self.logreg_dataset_class
            seed = self.seed
            l2_coeff_nr = self.l2_coeff_nr

            class NRAlgo(TorchNewtonRaphsonAlgo):
                def __init__(self):
                    super().__init__(
                        model=logreg_model,
                        batch_size=sys.maxsize,
                        criterion=nn.BCELoss(),
                        dataset=logreg_dataset_class,
                        seed=seed,
                        l2_coeff=l2_coeff_nr,
                    )

            self.nr_algo = NRAlgo()

            self.nr_strategy = NewtonRaphson(
                damping_factor=self.damping_factor_nr,
                algo=self.nr_algo,
                metric_functions=self.accuracy_metrics_dict,
            )
            self.propensity_model_strategy = self.nr_strategy

        # Allow for actual modifications of strategies if it is called afterwards
        if hasattr(self, "strategies"):
            self.strategies[0] = self.propensity_model_strategy
            self.strategies[1] = self.webdisco_strategy

    def reset_experiment(self):
        """Remove the propensity model just in case."""
        super().reset_experiment()
        if hasattr(self, "propensity_model"):
            self.propensity_model = None
        if hasattr(self, "propensity_models"):
            self.propensity_models = None
        self.logreg_model = LogisticRegressionTorch(self.ndim, self.torch_dtype)

    def fit(
        self,
        data: pd.DataFrame,
        targets: Optional[pd.DataFrame] = None,
        n_clients: Union[int, None] = None,
        split_method: Union[Callable, None] = None,
        split_method_kwargs: Union[Callable, None] = None,
        data_path: Union[str, None] = None,
        variance_method: Union[str, None] = None,
        n_bootstrap: Union[int, None] = None,
        bootstrap_seeds: Union[list[int], None] = None,
        bootstrap_function: Union[Callable, None] = None,
        dp_target_epsilon: Union[float, None] = None,
        dp_target_delta: Union[float, None] = None,
        dp_max_grad_norm: Union[float, None] = None,
        dp_propensity_model_training_params: Union[dict, None] = None,
        dp_propensity_model_optimizer_class: Union[Optimizer, None] = None,
        dp_propensity_model_optimizer_kwargs: Union[dict, None] = None,
        backend_type: str = "subprocess",
        urls: Union[list[str], None] = None,
        server_org_id: Union[str, None] = None,
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
        targets : Optional[pd.DataFrame], optional
            A dataframe with propensity score or nothing.
        nb_clients : Union[int, None], optional
            The number of clients used to split data across, by default None
        split_method : Union[Callable, None], optional
            How to split data across the nb_clients, by default None
        split_method_kwargs : Union[Callable, None], optional
            Argument of the function used to split data, by default None
        data_path : Union[str, None]
            Where to store the data on disk when backend is not remote.
        variance_method : `{"naive", "robust", "bootstrap"}`
            Method for estimating the variance, and therefore the p-value of the
            estimated treatment effect.
            * "naive": Inverse of the Fisher information.
            * "robust": The robust sandwich estimator [1] computed in FL thanks
                to FedECA. Useful when samples are reweighted.
            * "bootstrap": Bootstrap the given data by sampling each patient
              with replacement, each time estimate the treatment effect, then
              use all repeated estimations to compute the variance. The implementation
              is efficient in substra and shouldn't induce too much overhead.
            Defauts to naïve.
            [1] David A Binder. Fitting cox’s proportional hazards models from survey data. Biometrika, 79(1):139–147, 1992.  # noqa: E501
        n_bootstraps : Union[int, None]
            Number of bootstrap to be performed. If None will use
            len(bootstrap_seeds) instead. If bootstrap_seeds is given
            seeds those seeds will be used for the generation
            otherwise seeds are generated randomly.
        bootstrap_seeds : Union[list[int], None]
            The list of seeds used for bootstrapping random states.
            If None will generate n_bootstraps randomly, in the presence
            of both allways use bootstrap_seeds.
        bootstrap_function: Union[Callable, None]
            The bootstrap function to use for instance if it is necessary to mimic
            a global sampling.
        dp_target_epsilon: float
            The target epsilon for (epsilon, delta)-differential
            private guarantee. Defaults to None.
        dp_target_delta: float
            The target delta for (epsilon, delta)-differential
            private guarantee. Defaults to None.
        dp_max_grad_norm: float
            The maximum L2 norm of per-sample gradients;
            used to enforce differential privacy. Defaults to None.
        dp_propensity_model_optimizer_class: torch.optim.Optimizer
            The optimizer to use for the training of the propensity model.
            Defauts to Adam.
        dp_propensity_model_optimizer_class_kwargs: dict
            The params to give to optimizer class.
        dp_propensity_model_training_params: dict
            A dict with keys batch_size and num_updates for the DP-SGD training.
            Defaults to None.
        backend_type: str
            The backend to use for substra. Can be either:
            ["subprocess", "docker", "remote"]. Defaults to "subprocess".
        urls: Union[list[str], None]
            Urls corresponding to clients API if using remote backend_type.
            Defaults to None.
        server_org_id: Union[str, None]
            Url corresponding to server API if using remote backend_type.
            Defaults to None.
        tokens: Union[list[str], None]
            Tokens necessary to authenticate each client API if backend_type
            is remote. Defauts to None.
        """
        # Reset experiment so that it can fit on a new dataset
        self.reset_experiment()
        if backend_type != "remote" and (
            urls is not None or server_org_id is not None or tokens is not None
        ):
            logger.warning(
                "urls, server_org_id and tokens are ignored if backend_type is "
                "not remote; Make sure that you launched the fit with the right"
                " combination of parameters."
            )

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
            n_clients=n_clients,
            split_method=split_method,
            split_method_kwargs=split_method_kwargs,
            backend_type=backend_type,
            data_path=data_path,
            urls=urls,
            tokens=tokens,
        )
        if server_org_id is not None:
            # Curiously we don't need to identify the server with its own token
            # it's basically a passive entity
            kwargs_agg_node = {
                "organization_id": server_org_id,
            }
            self.aggregation_node = AggregationNode(**kwargs_agg_node)
        # Overwrites test_data_nodes
        if self.test_data_nodes is None:
            self.test_data_nodes = test_data_nodes
        else:
            raise ValueError(
                "You should not use the fit method if you already provided"
                " test_data_nodes"
            )

        # So there is a tension between every param is given at instantiation or
        # everything is given to fit
        dp_params_given = False
        for dp_param_name in [
            "dp_target_epsilon",
            "dp_target_delta",
            "dp_max_grad_norm",
            "dp_propensity_model_training_params",
            "dp_propensity_model_optimizer_class",
            "dp_propensity_model_optimizer_kwargs",
        ]:
            param = eval(dp_param_name)
            if param is not None:
                dp_params_given = True
                setattr(self, dp_param_name, param)

        if dp_params_given:
            # We need to reset the training mode more deeply
            self.set_propensity_model_strategy()

        if variance_method is not None:
            if variance_method != self.variance_method:
                self.variance_method = variance_method
            robust = variance_method == "robust"
            is_bootstrap = variance_method == "bootstrap"
            if robust != self.robust:
                self.robust = robust
            if is_bootstrap != self.is_bootstrap:
                # We need to reset the propensity model to have non zero
                # regularization
                self.set_propensity_model_strategy()

            if (
                is_bootstrap
                and is_bootstrap == self.is_bootstrap
                and any(
                    [
                        e is not None
                        for e in [n_bootstrap, bootstrap_seeds, bootstrap_function]
                    ]
                )
            ):
                raise ValueError("Not supported need refactor of fit/__init__")
            self.is_bootstrap = is_bootstrap

            if self.robust:
                assert self.is_bootstrap is False, "You can't use robust and bootstrap"

                class MockAlgo:
                    def __init__(self):
                        self.strategies = ["Robust Cox Variance"]

                mock_algo = MockAlgo()
                self.strategies.append(
                    RobustCoxVariance(
                        algo=mock_algo,
                        metric_functions={},
                    )
                )
                # We put WebDisco in "robust" mode in the sense that we ask it
                # to store all needed quantities for robust variance estimation
                self.strategies[
                    1
                ].algo._robust = True  # not sufficient for serialization
                # possible only because we added robust as a kwargs
                self.strategies[1].algo.kwargs.update({"robust": True})
                # We need this line for the zip to consider all 3
                # strategies
                self.num_rounds_list.append(sys.maxsize)
            else:
                self.strategies = self.strategies[:2]
                if variance_method == "bootstrap":
                    is_btst_strategy = [
                        hasattr(strat, "individual_strategies")
                        for strat in self.strategies
                    ]
                    if any(is_btst_strategy) and not all(is_btst_strategy):
                        assert (
                            "One of the strategy is a bootstrap strategy the"
                            "other is a regular strategy this should not happen"
                        )
                    if not any(is_btst_strategy):
                        # We only bootstrap the first strategy because
                        # for the glue to work for the second one we need to
                        # do it later and there is no 3rd one with bootstrap
                        if n_bootstrap is not None:
                            self.n_bootstrap = n_bootstrap
                        if bootstrap_seeds is not None:
                            self.bootstrap_seeds = bootstrap_seeds
                        if bootstrap_function is not None:
                            self.bootstrap_function = bootstrap_function

                        self.strategies = [
                            make_bootstrap_strategy(
                                self.strategies[0],
                                n_bootstrap=self.n_bootstrap,
                                bootstrap_seeds=self.bootstrap_seeds,
                                bootstrap_function=self.bootstrap_function,
                            )[0]
                        ] + self.strategies[1:]

        self.run(targets=targets)
        self.propensity_scores_, self.weights_ = self.compute_propensity_scores(data)

    def run(self, targets: Union[pd.DataFrame, None] = None):
        """Run the federated iptw algorithms."""
        del targets
        logger.info("Careful for now the argument target is ignored completely")
        # We first run the propensity model
        logger.info("Fitting the propensity model...")
        t1 = time.time()
        super().run(1)

        if not (self.simu_mode):
            self.check_cp_status()
            self.performances_propensity_model = pd.DataFrame(
                self.ds_client.get_performances(self.compute_plan_keys[0].key).dict()
            )
        else:
            self.performances_propensity_model = self.performances_strategies[0]
            logger.info(self.performances_propensity_model)
        t2 = time.time()
        self.propensity_model_fit_time = t2 - t1
        logger.info(f"Time to fit Propensity model {self.propensity_model_fit_time}s")
        logger.info("Finished, recovering the final propensity model from substra")
        # TODO to add the opportunity to use the targets you have to either:
        # give the full targets to every client as a kwargs of their Algo
        # so effectively one would need to reinstantiate algos objects or to
        # modify the API to do it in the run (cleaner)
        # or to rebuild the data on disk with an additional column that would be
        # the propensity score, aka rerun split_dataframes after having given it
        # an additional column and modify the algo so that it uses this column as
        # a score. Both schemes are quite cumbersome to implement.
        # We retrieve the model and pass it to the strategy
        # we run the IPTW Cox
        if not (self.simu_mode):
            algo = download_algo_state(
                client=self.ds_client
                if self.partner_client is None
                else self.partner_client,
                compute_plan_key=self.compute_plan_keys[0].key,
                round_idx=None,
            )
        else:
            # compute_plan_keys[0] is a tuple with
            # (scores, train_states, aggregate_states) in it
            train_state = get_simu_state_from_round(
                simu_memory=self.compute_plan_keys[0][1],
                client_id=self.ds_client.organization_info().organization_id,
                round_idx=None,  # Retrieve last round
            )
            algo = train_state.algo

        if self.variance_method != "bootstrap":
            self.propensity_model = algo.model
            # We give the trained model to WebDisco
            self.strategies[1].algo._propensity_model = self.propensity_model
            self.strategies[1].algo.kwargs.update(
                {"propensity_model": self.propensity_model}
            )
        else:
            self.propensity_models = [algo.model for algo in algo.individual_algos]
            # We give the trained models to WebDisco
            # Now we can create the bootstrap strategy as we have all the
            # propensity models
            self.strategies[1] = make_bootstrap_strategy(
                self.strategies[1],
                n_bootstrap=self.n_bootstrap,
                bootstrap_seeds=self.bootstrap_seeds,
                bootstrap_function=self.bootstrap_function,
                client_specific_kwargs=[
                    {"propensity_model": model} for model in self.propensity_models
                ],
            )[0]

        # We need to save intermediate outputs now
        for t in self.train_data_nodes:
            t.keep_intermediate_states = True

        logger.info("Fitting propensity weighted Cox model...")
        t1 = time.time()
        super().run(1)

        if not self.simu_mode:
            self.check_cp_status(idx=1)
        t2 = time.time()
        self.webdisco_fit_time = t2 - t1
        logger.info(f"Time to fit WebDisco {self.webdisco_fit_time}s")
        logger.info("Finished fitting weighted Cox model.")
        self.total_fit_time = self.propensity_model_fit_time + self.webdisco_fit_time
        self.print_summary()

    def print_summary(self):
        """Print a summary of the FedECA estimation."""
        assert (
            len(self.compute_plan_keys) == 2
        ), "You need to run the run method before getting the summary"
        logger.info("Evolution of performance of propensity model:")
        logger.info(self.performances_propensity_model)
        logger.info("Checking if the Cox model has converged:")
        self.get_final_cox_model()
        logger.info("Computing summary...")
        self.compute_summary()
        logger.info("Final partial log-likelihood:")
        self.log_likelihood_ = self.lls[0]
        logger.info(self.log_likelihood_)
        logger.info(self.results_)

    def get_final_cox_model(self):
        """Retrieve final cox model."""
        logger.info("Retrieving final hessian and log-likelihood")
        if not self.simu_mode:
            cp = self.compute_plan_keys[1].key
        else:
            cp = self.compute_plan_keys[1]

        (
            self.hessians,
            self.lls,
            self.final_params_list,
            self.computed_stds_list,
            self.global_robust_statistics,
        ) = get_final_cox_model_function(
            self.ds_client if self.partner_client is None else self.partner_client,
            cp,
            self.num_rounds_list[1],
            self.standardize_data,
            self.duration_col,
            self.event_col,
            simu_mode=self.simu_mode,
            robust=self.robust,
        )
        self.final_params_array = np.array(self.final_params_list)

    def compute_propensity_scores(self, data: pd.DataFrame):
        """Compute propensity scores and corresponding weights."""
        X = data.drop([self.duration_col, self.event_col, self.treated_col], axis=1)
        # dangerous but we need to do it
        string_columns = [col for col in X.columns if not (is_numeric_dtype(X[col]))]
        X = X.drop(columns=string_columns)
        Xprop = torch.from_numpy(np.array(X)).type(self.torch_dtype)
        with torch.no_grad():
            if hasattr(self, "propensity_model"):
                propensity_scores = self.propensity_model(Xprop)
            else:
                propensity_scores = self.propensity_models[0](Xprop)
                for model in self.propensity_models[1:]:
                    model.eval()
                    propensity_scores += model(Xprop)

        propensity_scores = propensity_scores.detach().numpy().flatten()
        weights = data[self.treated_col] * 1.0 / propensity_scores + (
            1 - data[self.treated_col]
        ) * 1.0 / (1.0 - propensity_scores)

        return np.array(propensity_scores), np.array(weights)

    def compute_summary(self, alpha=0.05):
        """Compute summary for a given threshold.

        Parameters
        ----------
        alpha: float, (default=0.05)
            Confidence level for computing CIs
        """
        if not (self.is_bootstrap):
            self.variance_matrix = -inv(self.hessians[0]) / np.outer(
                self.computed_stds_list[0], self.computed_stds_list[0]
            )
            if self.robust:
                assert self.global_robust_statistics
                beta = self.final_params_list[0]
                variance_matrix = self.variance_matrix
                global_robust_statistics = self.global_robust_statistics
                propensity_model = self.propensity_model
                duration_col = self.duration_col
                event_col = self.event_col
                treated_col = self.treated_col

                # no self attributes in this class !!!!!!
                class MyRobustCoxVarianceAlgo(RobustCoxVarianceAlgo):
                    def __init__(self, **kwargs):
                        super().__init__(
                            beta=beta,
                            variance_matrix=variance_matrix,
                            global_robust_statistics=global_robust_statistics,
                            propensity_model=propensity_model,
                            duration_col=duration_col,
                            event_col=event_col,
                            treated_col=treated_col,
                        )

                my_robust_cox_algo = MyRobustCoxVarianceAlgo()
                # Now we need to make sure strategy has the right algo
                self.strategies[2].algo = my_robust_cox_algo
                super().run(1)

                if not self.simu_mode:
                    self.check_cp_status(idx=2)
                    # We assume that the builder of tasks is the
                    self.variance_matrix = get_outmodel_function(
                        "Aggregating Qk into Q",
                        self.ds_client,
                        compute_plan_key=self.compute_plan_keys[2].key,
                        idx_task=0,
                    )

                else:
                    # Awful but hard to hack better
                    # compute_plan_keys[2] is a tuple with
                    # (scores, train_states, aggregate_states) in it
                    self.variance_matrix = sum(
                        [
                            e.algo._client_statistics["Qk"]
                            for e in self.compute_plan_keys[2][1].state
                        ]
                    )

        else:
            # We directly estimate the variance matrix in the bootstrap case
            # we drop the first run which is not bootstrapped
            # bias=True to match the pooled implementation
            self.variance_matrix = np.cov(
                self.final_params_array[1:], rowvar=False, bias=True
            ).reshape(
                (self.final_params_array.shape[1], self.final_params_array.shape[1])
            )

        # The final params vector is the params wo bootstrap
        if self.final_params_array.ndim == 2:
            final_params = self.final_params_array[0]
        else:
            final_params = self.final_params_array

        summary = compute_summary_function(final_params, self.variance_matrix, alpha)

        summary["exp(coef)"] = np.exp(summary["coef"])
        summary["exp(coef) lower 95%"] = np.exp(summary["coef lower 95%"])
        summary["exp(coef) upper 95%"] = np.exp(summary["coef upper 95%"])

        self.results_ = summary.copy()


class LogisticRegressionTorch(nn.Module):
    """Pytorch logistic regression class."""

    def __init__(self, ndim, torch_dtype=torch.float64):
        """Initialize Logistic Regression model in PyTorch.

        Parameters
        ----------
        ndim : int
            Number of input dimensions.
        torch_dtype : torch.dtype, optional
            Data type for PyTorch tensors, by default torch.float64.
        """
        self.torch_dtype = torch_dtype
        self.ndim = ndim
        super(LogisticRegressionTorch, self).__init__()
        self.fc1 = nn.Linear(self.ndim, 1).to(self.torch_dtype)
        # Zero-init as in sklearn
        self.fc1.weight.data.fill_(0.0)
        self.fc1.bias.data.fill_(0.0)

    def forward(self, x, eval=False):
        """Perform a forward pass through the Logistic Regression model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ndim).
        eval : bool, optional
            Set to True during evaluation, by default False.

        Returns
        -------
        torch.Tensor
            Predicted probabilities after passing through sigmoid activation.
        """
        x = self.fc1(x)
        return torch.sigmoid(x)
