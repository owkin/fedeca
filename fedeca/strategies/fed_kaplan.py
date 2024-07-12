from typing import List, Optional, Union

import numpy as np
from substrafl import ComputePlanBuilder
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.nodes import AggregationNodeProtocol, TrainDataNodeProtocol
from substrafl.remote import remote, remote_data
from torch import nn

from fedeca.utils.survival_utils import (
    aggregate_events_statistics,
    compute_events_statistics,
    km_curve,
)


class FedKaplan(ComputePlanBuilder):
    def __init__(
        self,
        treated_col: Union[None, str] = None,
        propensity_model: Union[None, nn.Module] = None,
    ):
        """FedKaplan strategy. This class implements a federated version of Kaplan Meier
        estimates.

        Parameters
        ----------
        treated_col : Union[None, str], optional
            The column describing the treatment, by default None
        propensity_model : Union[None, nn.Module], optional
            _description_, by default None
        """

        super().__init__()
        assert not (
            (treated_col is None) and (propensity_model is not None)
        ), "if propensity model is provided, treatment_col should be provided as well"
        self._treated_col = treated_col
        self._propensity_model = propensity_model

    def build_compute_plan(
        self,
        train_data_nodes: Optional[List[TrainDataNodeProtocol]],
        aggregation_node: Optional[List[AggregationNodeProtocol]],
        evaluation_strategy: Optional[EvaluationStrategy],
        num_rounds: Optional[int],
        clean_models: Optional[bool] = True,
    ):

        assert len(self.local_computations) == len(self.aggregations)
        agg_shared_state = None
        step_idx = 0

        shared_states = []
        for node in train_data_nodes:
            # define composite tasks (do not submit yet)
            # for each composite task give description of
            # Algo instead of a key for an algo
            _, next_shared_state = node.update_states(
                self.compute_events_statistics(
                    node.data_sample_keys,
                    shared_state=agg_shared_state,
                    _algo_name="Compute Events Statistics",
                ),
                local_state=None,
                round_idx=step_idx,
                authorized_ids=set([node.organization_id]),
                aggregation_id=aggregation_node.organization_id,
                clean_models=False,
            )
            # keep the states in a list: one/organization
            shared_states.append(next_shared_state)

        agg_shared_state = aggregation_node.update_states(
            self.compute_agg_km_curve(
                shared_states=shared_states,
                _algo_name="Aggregate Events Statistics",
            ),
            round_idx=step_idx,
            authorized_ids=set(
                train_data_node.organization_id for train_data_node in train_data_nodes
            ),
            clean_models=False,
        )

    @remote_data
    def compute_events_statistics(
        self,
        datasamples,
        shared_state=None,
    ):
        """Computes events statistics for a subset of data.
        group_triplet : HyperParameter, optional
            Triplet describing the subpopulation to use, by default None.
            This triplet has the format: `(column_index, op, cutoff_val)`
            where `column_index` is the index of the column on which the
            selection should take place, `op` the operator to use, as a
            string (one of `"<"`, `"<="`, `">"`, `">="`), and `cutoff_val`
            the value to cut. If the dataset is stored in matrix `X`, this
            means we select samples satisfying
            `X[:, column_index] op cutoff_val`

        Returns
        -------
        Placeholder
            Method output or placeholder thereof.
        """
        # we only use survival times
        X, y, weights = self.compute_X_y_and_propensity_weights(datasamples, None)
        weighted_X = weights * X
        # retrieve times and events
        times = np.abs(y)
        events = y > 0
        return compute_events_statistics(times, events)

    @remote_data
    def compute_km_curve(
        self,
        datasamples,
    ):
        """Computes Kaplan-Meier curve for a subset of data.

        Returns
        -------
        Placeholder
            Method output or placeholder thereof.
        """
        t, n, d = self.compute_events_statistics(
            datasamples=datasamples,
            _skip=True,
        )
        return km_curve(t, n, d)

    @remote
    def aggregate_events_statistics(
        self,
        shared_states,
    ):
        """Aggregates events statistics for a subset of data."""
        return aggregate_events_statistics(shared_states)

    @remote
    def compute_agg_km_curve(self, shared_states):
        """Computes the aggregated Kaplan-Meier curve."""
        t_agg, n_agg, d_agg = aggregate_events_statistics(shared_states)
        return km_curve(t_agg, n_agg, d_agg)
