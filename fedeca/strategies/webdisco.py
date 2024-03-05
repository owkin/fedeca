"""File for webdisco strategy."""
from copy import deepcopy
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from substrafl.algorithms.algo import Algo
from substrafl.nodes.aggregation_node import AggregationNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.nodes.references.shared_state import SharedStateRef
from substrafl.nodes.test_data_node import TestDataNode
from substrafl.nodes.train_data_node import TrainDataNode
from substrafl.remote import remote

# from substrafl.schemas import WebDiscoAveragedStates
# from substrafl.schemas import WebDiscoSharedState
from substrafl.strategies.strategy import Strategy

from fedeca.utils.moments_utils import aggregation_mean, compute_centered_moment


class StrategyName(str, Enum):
    """Class for the strategy name."""

    WEBDISCO = "WebDisco"


class WebDisco(Strategy):
    """WebDisco strategy class.

    It can only be used with traditional Cox models on pandas.DataFrames.
    This strategy is one of its kind because it can only be used with
    Linear CoxPH models defined in fedeca.utils.survival_utils. Therefore all models are
    initialized with zeroed weights (as in lifelines), tested and we cover all possible
    use cases with the dtype and ndim arguments. This strategy splits the computations
    of gradient and Hessian between workers to compute a centralized batch Newton-
    Raphson update on Breslow's partial log-likelihod (to handle tied events it uses
    Breslow's approximation unlike lifelines which uses Efron's by default but Efron is
    not separable). This strategy uses lifeline's adaptive step-size to converge faster
    starting from initial_ste_size and use lifelines safe way of inverting the hessian.
    As lifelines standardizes the data by default we allow the user to do it optionally.

    Reference
    ----------
    - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5009917/

    Parameters
    ----------
    statistics_computed: bool,
        If the statistics that we can find in each gradient, hessian are already
        computed and given as attribute to the server or not.
    initial_step_size: float, otional
        The initial step size of the Newton-Raphson algorithm at the server side.
        The following steps will use lifelines heuristics to adapt the step-size.
        Defaults to 0.9.
    tol: float, optional
        Capping every division to avoid dividing by 0. Defaults to 1e-16.
    standardize_data: bool,
        Whether or not to standardize the data before comuting updates.
        Defaults to False.
    penalizer: float, optional
        Add a regularizer in case of ill-conditioned hessians, which happen quite
        often with large covariates.
        Defaults to 0.
    l1_ratio: float, optional
        When using a penalizer the ratio between L1 and L2 regularization as in
        sklearn.
        Defaults to 0.
    """

    def __init__(
        self,
        algo: Algo,
        metric_functions: Optional[
            Union[Dict[str, Callable], List[Callable], Callable]
        ] = None,
        standardize_data: bool = True,
        tol: float = 1e-16,
    ):
        """Initialize the Webdisco class.

        Parameters
        ----------
        algo: Algo
            Algorithm needed to perform the optimization.
        metric_functions (Optional[Union[Dict, List, Callable]]):
            list of Functions that implement the different metrics.
            If a Dict is given, the keys will be used to register
            the result of the associated function. If a Function or
            a List is given, function.__name__ will be used to store
            the result.
        standardize_data: bool,
            Whether or not to standardize each features
            (if True involves more tasks in order to compute
            global means and stds)
        tol: float,
            Epsilon used to ensure no ZeroDivision Errors due to finite
            numerical precision.
        """
        # !!! You actually need to pass all arguments explicitly through this init
        # function so that kwargs is instantiated with the correct arguments !!!
        super().__init__(
            algo=algo,
            metric_functions=metric_functions,
            standardize_data=standardize_data,
            tol=tol,
        )

        # States
        self._local_states: Optional[List[LocalStateRef]] = None
        self._shared_states: Optional[List[SharedStateRef]] = None

        self._standardize_data = standardize_data
        self._tol = tol
        self._survival_statistics_computed = False
        self._gs_statistics_given = False
        self._server_state = None
        self.n_clients = None
        self.count = 1

    @property
    def name(self) -> StrategyName:
        """The name of the strategy.

        Returns
        -------
        StrategyName: Name of the strategy
        """
        return StrategyName.WEBDISCO

    def build_compute_plan(
        self,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: Optional[List[AggregationNode]],
        evaluation_strategy,
        num_rounds: int,
        clean_models: Optional[bool],
    ):
        """Build the computation graph of the strategy.

        It removes initialization round,
        which is useless in this case as all models start at 0.

        Parameters
        ----------
        train_data_nodes: typing.List[TrainDataNode],
            list of the train organizations
        aggregation_node: typing.Optional[AggregationNode],
            aggregation node, necessary for centralized strategy, unused otherwise
        evaluation_strategy: Optional[EvaluationStrategy],
          When and how to compute performance.
        num_rounds: int,
            The number of rounds to perform.
        clean_models: bool (default=True),
            Clean the intermediary models on the Substra platform.
            Set it to False if you want to download or re-use
            intermediary models. This causes the disk space to
            fill quickly so should be set to True unless needed.
            Defaults to True.
        """
        additional_orgs_permissions = (
            evaluation_strategy.test_data_nodes_org_ids
            if evaluation_strategy is not None
            else set()
        )
        # create computation graph.
        for round_idx in range(0, num_rounds + 1):
            self.perform_round(
                train_data_nodes=train_data_nodes,
                aggregation_node=aggregation_node,
                additional_orgs_permissions=additional_orgs_permissions,
                round_idx=round_idx,
                clean_models=clean_models,
            )

            if evaluation_strategy is not None and next(evaluation_strategy):
                self.perform_evaluation(
                    train_data_nodes=train_data_nodes,
                    test_data_nodes=evaluation_strategy.test_data_nodes,
                    round_idx=round_idx,
                )

    def _global_standardization(
        self,
        local_computation_fct,
        aggregation_fct,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        clean_models: bool,
    ):
        local_moments = []
        for node in train_data_nodes:
            # define composite tasks (do not submit yet)
            # for each composite task give description of
            # Algo instead of a key for an algo
            _, local_moment = node.update_states(
                local_computation_fct(
                    node.data_sample_keys,
                    shared_state=None,
                    _algo_name=local_computation_fct.__doc__.split("\n")[0],
                ),
                local_state=None,
                round_idx=self.count,
                authorized_ids=set([node.organization_id]),
                aggregation_id=aggregation_node.organization_id,
                clean_models=False,
            )
            # keep the states in a list: one/organization
            local_moments.append(local_moment)

        aggregated_moments = aggregation_node.update_states(
            aggregation_fct(
                shared_states=local_moments,
                _algo_name=aggregation_fct.__doc__.split("\n")[0],
            ),
            round_idx=self.count,
            authorized_ids=set(
                train_data_node.organization_id for train_data_node in train_data_nodes
            ),
            clean_models=clean_models,
        )
        self.count += 1
        return aggregated_moments

    def _global_statistics(
        self,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        clean_models: bool,
    ):
        self._local_states = []
        survival_statistics_list = []
        for node in train_data_nodes:
            # define composite tasks (do not submit yet)
            # for each composite task give description of
            # Algo instead of a key for an algo
            local_state, next_shared_state = node.update_states(
                self.algo._compute_local_constant_survival_statistics(
                    node.data_sample_keys,
                    shared_state=self._aggregated_moments,
                    _algo_name="Compute Local Statistics",
                ),
                local_state=None,
                round_idx=self.count,
                authorized_ids=set([node.organization_id]),
                aggregation_id=aggregation_node.organization_id,
                clean_models=clean_models,
            )
            # keep the states in a list: one/organization
            survival_statistics_list.append(next_shared_state)
            self._local_states.append(local_state)

        global_survival_statistics = aggregation_node.update_states(
            self._compute_global_survival_statistics(
                shared_states=survival_statistics_list,
                _algo_name="Compute global statistics from local quantities",
            ),
            round_idx=self.count,
            authorized_ids=set(
                train_data_node.organization_id for train_data_node in train_data_nodes
            ),
            clean_models=clean_models,
        )
        self.count += 1
        self._survival_statistics_computed = True
        return global_survival_statistics

    def perform_round(
        self,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        round_idx: int,
        clean_models: bool,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """Perform one round of webdisco.

        One round of the WebDisco strategy consists in:
            - optionally compute global means and stds for all features if
              standardize_data is True
            - compute global survival statistics that will be reused at each round
            - build building blocks of the gradient and hessian based on global risk
              sets
            - perform a Newton-Raphson update on each train data nodes
        Parameters
        ----------
        train_data_nodes: typing.List[TrainDataNode],
            List of the nodes on which to train
        aggregation_node: AggregationNode
            node without data, used to perform operations on the
            shared states of the models
        round_idx :int,
            Round number, it starts at 0.
        clean_models: bool,
            Clean the intermediary models of this round on the
            Substra platform. Set it to False if you want to
            download or re-use intermediary models. This causes the
            disk space to fill quickly so should be set to True unless needed.
        additional_orgs_permissions: typing.Optional[set],
            Additional permissions to give to the model outputs after training,
            in order to test the model on an other organization.
        """
        if aggregation_node is None:
            raise ValueError("In WebDisco strategy aggregation node cannot be None")

        # Since algo and FL strategies are split we need to add this ugly assert here,
        # note that we could force the algo.
        # params to respect the strategies and the other way around
        assert (
            self.algo._standardize_data == self._standardize_data
        ), f"""Algo and strategy standardize_data param differ
            {self.algo._standardize_data}!={self._standardize_data}"""

        # All models are initialized at
        if self._standardize_data and (not hasattr(self, "_aggregated_moments")):
            for _, (local_computation_fct, aggregation_fct) in enumerate(
                zip([self.algo.local_uncentered_moments], [self.aggregate_moments])
            ):
                self._aggregated_moments = self._global_standardization(
                    local_computation_fct=local_computation_fct,
                    aggregation_fct=aggregation_fct,
                    train_data_nodes=train_data_nodes,
                    aggregation_node=aggregation_node,
                    clean_models=clean_models,
                )

        else:
            self._aggregated_moments = None

        if not (self._survival_statistics_computed):
            # Uses self._aggregated_moments internally
            global_survival_statistics = self._global_statistics(
                train_data_nodes=train_data_nodes,
                aggregation_node=aggregation_node,
                clean_models=clean_models,
            )
            # !!! The reason we are doing compute_local_phi_stats only once is
            # subtle this is because to optimize we do it in train

            risk_phi_stats_list = []
            for i, node in enumerate(train_data_nodes):
                # define composite tasks (do not submit yet)
                # for each composite task give description of Algo instead of a key
                # for an algo
                local_state, risk_phi_stats = node.update_states(
                    self.algo.compute_local_phi_stats(  # type: ignore
                        node.data_sample_keys,
                        shared_state=global_survival_statistics,
                        _algo_name=f"Compute gradients and hessian bricks locally using algo {self.algo.__class__.__name__}",  # noqa: E501
                    ),
                    local_state=self._local_states[i],
                    round_idx=self.count,
                    authorized_ids=set([node.organization_id])
                    | additional_orgs_permissions,
                    aggregation_id=aggregation_node.organization_id,
                    clean_models=clean_models,
                )
                # keep the states in a list: one/organization
                risk_phi_stats_list.append(risk_phi_stats)
                self._local_states[i] = local_state

            self._shared_states = risk_phi_stats_list

        # Now this assumes that in self._shared_states we have both:
        # - the current risk_phi_stats computed from train
        # - global_survival_statistics that should be perpetually given to
        # the server which is stateless
        # - the server state (parameters, log-likelihood and convergence stuff)
        self._global_gradient_and_hessian = aggregation_node.update_states(
            self._build_global_gradient_and_hessian(
                shared_states=self._shared_states,
                _algo_name="Compute gradient and hessian",
            ),
            round_idx=self.count,
            authorized_ids=set(
                train_data_node.organization_id for train_data_node in train_data_nodes
            ),
            clean_models=clean_models,
        )
        self.count += 1

        # We now need local states cause we'll update the model
        next_local_states = []
        next_shared_states = []

        for i, node in enumerate(train_data_nodes):
            # define composite tasks (do not submit yet)
            # for each composite task give description of Algo instead of a key for an
            # algo
            next_local_state, local_risk_phi_stats = node.update_states(
                # This does a compute_local_phi_stats with skip=True this explains
                # why there is no
                # need to call it explicitly after self._survival_statistics_computed
                # becomes True
                self.algo.train(  # type: ignore
                    node.data_sample_keys,
                    shared_state=self._global_gradient_and_hessian,
                    _algo_name=f"Training with {self.algo.__class__.__name__}",  # noqa: E501
                ),
                local_state=self._local_states[i],
                round_idx=self.count,
                authorized_ids=set([node.organization_id])
                | additional_orgs_permissions,
                aggregation_id=aggregation_node.organization_id,
                clean_models=clean_models,
            )
            # keep the states in a list: one/node
            next_local_states.append(next_local_state)
            next_shared_states.append(local_risk_phi_stats)

        self.count += 1

        # Now that the models are updated we'll use them in the next round
        self._local_states = next_local_states
        self._shared_states = next_shared_states

    @remote
    def _compute_global_survival_statistics(self, shared_states):
        """Aggregate different needed statistics.

        Compute aggregated statistics such as distinct event times, sum of
        covariates occuring on each event, total weights of parameters on all events,
        etc.

        Parameters
        ----------
        shared_states : list[dict]
            A list of dicts of covariate statistics and distinct event times from each
            center.
            The expected keys are 'sum_features_on_events' which is a vector of the same
            shape as a feature,
            'distinct_event_times', which is a list with distinct event times.
            'number_events_by_time' which is a list for all events with the number of
            features that have the same event.
            'total_number_of_samples', which is an integer,
            'weights_counts_on_events', the weights of all covariates on
            each event match number_events_by_time if no propensity model is used.


        Returns
        -------
        list
            The global list of distinct values
        """
        global_sum_features_on_events = np.zeros_like(
            shared_states[0]["sum_features_on_events"]
        )

        # find out all distinct values while avoiding duplicates
        distinct_event_times = []
        for ls_and_dv in shared_states:
            distinct_event_times += ls_and_dv["distinct_event_times"]
        distinct_event_times = list(set(distinct_event_times))
        distinct_event_times.sort()
        # count them
        num_global_event_times = len(distinct_event_times)
        # aggregate statistics by suming
        for ls_and_dv in shared_states:
            global_sum_features_on_events += ls_and_dv["sum_features_on_events"]
        # Count the number of tied event times for each client
        list_number_events_by_time = []
        total_number_samples = sum(
            [ls_and_dv["total_number_samples"] for ls_and_dv in shared_states]
        )

        # Very weird to double check that it cannot be written in a more readable way
        for ls_and_dv in shared_states:
            global_ndt = []
            for i, e in enumerate(distinct_event_times):
                if e in ls_and_dv["distinct_event_times"]:
                    idx = ls_and_dv["distinct_event_times"].index(e)
                    global_ndt.append(ls_and_dv["number_events_by_time"][idx])
                else:
                    global_ndt.append(0)
            list_number_events_by_time.append(global_ndt)

        # We add what should amount at number events by time if weights=1
        weights_counts_on_events = []
        for d in distinct_event_times:
            weights_counts_on_event = 0.0
            for ls_and_dv in shared_states:
                if d in ls_and_dv["distinct_event_times"]:
                    idx = ls_and_dv["distinct_event_times"].index(d)
                    weights_counts_on_event += ls_and_dv["weights_counts_on_events"][
                        idx
                    ]
            weights_counts_on_events.append(weights_counts_on_event)

        results = {}
        results["global_survival_statistics"] = {}
        results["global_survival_statistics"][
            "distinct_event_times"
        ] = distinct_event_times
        results["global_survival_statistics"][
            "global_sum_features_on_events"
        ] = global_sum_features_on_events
        results["global_survival_statistics"][
            "list_number_events_by_time"
        ] = list_number_events_by_time
        results["global_survival_statistics"][
            "num_global_events_time"
        ] = num_global_event_times
        results["global_survival_statistics"][
            "total_number_samples"
        ] = total_number_samples
        results["global_survival_statistics"][
            "weights_counts_on_events"
        ] = weights_counts_on_events
        results["moments"] = shared_states[0]["moments"]

        return results

    @remote
    def _build_global_gradient_and_hessian(
        self,
        shared_states,
    ):
        r"""Compute global gradient and Hessian.

        Use the gradient and hessian local blocks from clients to compute
        the global gradient and hessian and use them to compute Newton-Raphson
        update.
        Regarding the use of an L1 regularization we match lifelines and use the
        following coefficient by coefficient approximation of the absolute value (see
        https://www.cs.ubc.ca/sites/default/files/tr/2009/TR-2009-19_0.pdf
        page 7 equation 3) with t the index of the round:
        .. math::
            |x| = (x)_{+} + (-x)_{+}
            (x)_{+} \\approx x + \\frac{1}{\\alpha} \\cdot \\log(1 + \\exp(-\\alpha x))
            |x| \\approx \\frac{1}{\\alpha} \\cdot \\left(\\log(1 + \\exp(-\\alpha x) + \\log(1 + \\exp(\\alpha x)\\right)  # noqa: E501
            \\alpha = 1.3^{t}

        Parameters
        ----------
        risk_phi_stats_list : list
            A list of blocks necessary to compute the gradients and hessian.
        save_hessian_and_gradients : bool, optional
            Wether or not to save the value of the gradient and the hessian as attribute
            of the server, by default False
        forced_step_size: float, optional
            If not none, force the step size to be equal to the given value. Default
            None.
            Useful for tests.

        Returns
        -------
        list
            list of size 1 with NR update for the weights
        """
        # Server is stateless need to continuously feed it with
        # global_survival_statistics

        global_survival_statistics = shared_states[0]["global_survival_statistics"]
        # It is important to use deepcopy to avoid side effect
        # Otherwise, the value of self.global_sum_features_on_events will change
        # This is already weighted
        gradient = deepcopy(global_survival_statistics["global_sum_features_on_events"])
        ll = 0.0
        try:
            gradient_shape = [e for e in gradient.shape if e > 1][0]
        except IndexError:
            gradient_shape = 1

        risk_phi_stats_list = [e["local_phi_stats"] for e in shared_states]
        risk_phi_list = [e["risk_phi"] for e in risk_phi_stats_list]
        risk_phi_x_list = [e["risk_phi_x"] for e in risk_phi_stats_list]
        risk_phi_x_x_list = [e["risk_phi_x_x"] for e in risk_phi_stats_list]

        distinct_event_times = global_survival_statistics["distinct_event_times"]

        hessian = np.zeros((gradient_shape, gradient_shape))

        # Needed for robust estimation of SE
        global_risk_phi_list = []
        global_risk_phi_x_list = []
        # We first sum over each event
        for idxd, _ in enumerate(distinct_event_times):
            # This factor amounts to d_i the number of events per time i if no weights
            # otherwise it's the sum of the score of all d_i events
            weighted_average = global_survival_statistics["weights_counts_on_events"][
                idxd
            ]

            # We initialize both tensors at zeros for numerators (all denominators are
            # scalar)
            numerator = np.zeros(risk_phi_x_list[0][0].shape)
            # The hessian has several terms due to deriving quotient of functions u/v
            first_numerator_hessian = np.zeros((gradient_shape, gradient_shape))
            denominator = 0.0
            if np.allclose(weighted_average, 0.0):
                continue
            for i in range(len(risk_phi_stats_list)):
                numerator += risk_phi_x_list[i][idxd]
                denominator += risk_phi_list[i][idxd]
                first_numerator_hessian += risk_phi_x_x_list[i][idxd]

            global_risk_phi_list.append(denominator)
            global_risk_phi_x_list.append(numerator)
            # denominator being a sum of exponential it's always positive

            assert denominator >= 0.0, "the sum of exponentials is negative..."
            denominator = max(denominator, self._tol)
            denominator_squared = max(denominator**2, self._tol)
            c = numerator / denominator
            ll -= weighted_average * np.log(denominator)
            gradient -= weighted_average * np.squeeze(c)
            hessian -= weighted_average * (
                (first_numerator_hessian / denominator)
                - (np.multiply.outer(numerator, numerator) / denominator_squared)
            )

        return {
            "hessian": hessian,
            "gradient": gradient,
            "second_part_ll": ll,
            "gradient_shape": gradient_shape,
            "global_risk_phi_list": global_risk_phi_list,
            "global_risk_phi_x_list": global_risk_phi_x_list,
        }

    @remote
    def aggregate_moments(self, shared_states):
        """Compute the global centered moments given the local results.

        Parameters
        ----------
        shared_states : List
            List of results (local_m1, local_m2, n_samples) from training nodes.

        Returns
        -------
        dict
            Global results to be shared with train nodes via shared_state.
        """
        # aggregate the moments.

        tot_uncentered_moments = [
            aggregation_mean(
                [s[f"moment{k}"] for s in shared_states],
                [s["n_samples"] for s in shared_states],
            )
            for k in range(1, 2 + 1)
        ]
        n_samples = sum([s["n_samples"].iloc[0] for s in shared_states])
        results = {
            f"global_centered_moment_{k}": compute_centered_moment(
                tot_uncentered_moments[:k]
            )
            for k in range(1, 2 + 1)
        }
        results.update(
            {
                f"global_uncentered_moment_{k+1}": moment
                for k, moment in enumerate(tot_uncentered_moments)
            }
        )
        results.update({"total_n_samples": n_samples})
        return results

    def perform_evaluation(
        self,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):
        """Evaluate function for test_data_nodes on which the model have been trained
        on.

        Parameters
        ----------
        test_data_nodes: List[TestDataNode]),
            test data nodes to intersect with train data nodes to evaluate the
            model on.
        train_data_nodes: List[TrainDataNode],
            train data nodes the model has been trained on.
        round_idx: int,
            round index.

        Raises
        ------
            NotImplementedError: Cannot test on a node we did not train on for now.
        """
        for test_data_node in test_data_nodes:
            matching_train_nodes = [
                train_node
                for train_node in train_data_nodes
                if train_node.organization_id == test_data_node.organization_id
            ]
            if len(matching_train_nodes) == 0:
                node_index = 0
            else:
                node_index = train_data_nodes.index(matching_train_nodes[0])

            assert (
                self._local_states is not None
            ), "Cannot predict if no training has been done beforehand."
            local_state = self._local_states[node_index]

            test_data_node.update_states(
                operation=self.evaluate(
                    data_samples=test_data_node.data_sample_keys,
                    _algo_name=f"Evaluating with {self.algo.__class__.__name__}",
                ),
                traintask_id=local_state.key,
                round_idx=round_idx,
            )  # Init state for testtask
