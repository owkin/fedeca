"""Estimate the variance for mispecified Cox models."""
from typing import List, Optional

import numpy as np
from substrafl.nodes import AggregationNode, TrainDataNode
from substrafl.remote import remote
from substrafl.strategies.strategy import Strategy


class RobustCoxVariance(Strategy):
    """Launch robust variance estimation for cox models."""

    def __init__(self, algo, metric_functions: Optional = None):
        """Init robust cox variance estimation.

        Parameters
        ----------
        algo : RobustCoxVarianceAlgo
            An instance of RobustCoxVarianceAlgo.
        metric_functions (Optional[Union[Dict, List, Callable]]):
            list of Functions that implement the different metrics.
            If a Dict is given, the keys will be used to register
            the result of the associated function. If a Function
            or a List is given, function.__name__ will be used
            to store the result.
        """
        super().__init__(algo=algo, metric_functions=metric_functions)

        # States
        self._local_states: Optional[List[LocalStateRef]] = None
        self._shared_states: Optional[List[SharedStateRef]] = None

    # We have to have instantiated name, perform_evaluation and performm_round
    @property
    def name(self):
        """Set strategy name.

        Returns
        -------
        StrategyName
            Name of the strategy
        """
        return "Robust Cox Variance"

    def perform_evaluation(self):
        """Do nothing.

        Only there so that the strategy is recognized as such by substrafl.
        """
        pass

    def perform_round(self):
        """Do nothing.

        Only there so that the strategy is recognized as such by substrafl.
        """
        pass

    @remote
    def sum(self, shared_states: List[np.ndarray]):
        """Compute sum of Qks.

        Parameters
        ----------
        shared_states : List[np.ndarray]
            list of dictionaries containing Qk.

        Returns
        -------
        np.ndarray
            Q matrix.
        """
        return sum(shared_states)

    def build_compute_plan(
        self,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
        num_rounds=None,
        evaluation_strategy=None,
        clean_models=False,
    ):
        """Build compute plan.

        Method to build and link the different computations to execute with each
        other. We will use the ``update_state``method of the nodes given as input to
        choose which method to apply. For our example, we will only use TrainDataNodes
        and AggregationNodes.

        Parameters
        ----------
        train_data_nodes : List[TrainDataNode])
            Nodes linked to the data
            samples on which to compute analytics.
        aggregation_node : AggregationNode)
            Node on which to compute the
            aggregation of the analytics extracted from the train_data_nodes.
        num_rounds : Optional[int]
            Num rounds to be used to iterate on
            recurrent part of the compute plan. Defaults to None.
        evaluation_strategy : Optional[substrafl.EvaluationStrategy]
            Object storing the TestDataNode. Unused in this example. Defaults to None.
        clean_models : bool
            Clean the intermediary models of this round on
            the Substra platform. Default to False.
        """
        if self.algo is None:
            raise ValueError(
                "You should initialize the algo of this strategy with a"
                " RobustCoxVarianceAlgo."
            )

        qk_list = []

        for node in train_data_nodes:
            # Call local_first_order_computation on each train data node
            next_local_state, next_shared_state = node.update_states(
                self.algo.local_q_computation(
                    node.data_sample_keys,
                    shared_state=None,
                    _algo_name=f"Computing local Qk {self.__class__.__name__}",
                ),
                local_state=None,
                round_idx=0,
                authorized_ids=set([node.organization_id]),
                aggregation_id=aggregation_node.organization_id,
                clean_models=False,
            )

            # All local analytics are stored in the first_order_shared_states,
            # given as input the the aggregation method.
            qk_list.append(next_shared_state)
            # Just in case
            # self._local_states.append(next_local_state)

        # Call the aggregation method on the first_order_shared_states
        self.Q = aggregation_node.update_states(
            self.sum(
                shared_states=qk_list,
                _algo_name="Aggregating Qk into Q",
            ),
            round_idx=0,
            authorized_ids=set(
                [
                    train_data_node.organization_id
                    for train_data_node in train_data_nodes
                ]
            ),
            clean_models=False,
        )
