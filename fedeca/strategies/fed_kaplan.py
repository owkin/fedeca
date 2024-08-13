"""Compute federated Kaplan-Meier estimates."""
import pickle as pk
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from substrafl import ComputePlanBuilder
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.nodes import AggregationNodeProtocol, TrainDataNodeProtocol
from substrafl.remote import remote, remote_data
from torch import nn

from fedeca.utils.survival_utils import (
    aggregate_events_statistics,
    build_X_y_function,
    compute_events_statistics,
    compute_X_y_and_propensity_weights_function,
    km_curve,
)


class FedKaplan(ComputePlanBuilder):
    """Instantiate a federated version of Kaplan Meier estimates.

    Parameters
    ----------
    ComputePlanBuilder : _type_
        _description_
    """

    def __init__(
        self,
        duration_col: str,
        event_col: str,
        treated_col: str,
        client_identifier: str,
        propensity_model: Union[None, nn.Module] = None,
        tol: float = 1e-16,
    ):
        """Implement a federated version of Kaplan Meier estimates.

        This code is an adaptation of a previous implementation by Constance Beguier.

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
        self._duration_col = duration_col
        self._event_col = event_col
        self._treated_col = treated_col
        self._propensity_model = propensity_model
        self._client_identifier = client_identifier
        self._target_cols = [self._duration_col, self._event_col]
        self._tol = tol
        self.statistics_result = None
        self.kwargs["duration_col"] = duration_col
        self.kwargs["event_col"] = event_col
        self.kwargs["treated_col"] = treated_col
        self.kwargs["propensity_model"] = propensity_model
        self.kwargs["client_identifier"] = client_identifier
        self.kwargs["tol"] = tol

    def build_compute_plan(
        self,
        train_data_nodes: Optional[List[TrainDataNodeProtocol]],
        aggregation_node: Optional[List[AggregationNodeProtocol]],
        evaluation_strategy: Optional[EvaluationStrategy],
        num_rounds: Optional[int],
        clean_models: Optional[bool] = True,
    ):
        """Build the computation plan.

        Parameters
        ----------
        train_data_nodes : Optional[List[TrainDataNodeProtocol]]
            _description_
        aggregation_node : Optional[List[AggregationNodeProtocol]]
            _description_
        evaluation_strategy : Optional[EvaluationStrategy]
            _description_
        num_rounds : Optional[int]
            _description_
        clean_models : Optional[bool], optional
            _description_, by default True
        """
        del num_rounds
        del evaluation_strategy
        del clean_models
        shared_states = []
        for node in train_data_nodes:
            # define composite tasks (do not submit yet)
            # for each composite task give description of
            # Algo instead of a key for an algo
            _, next_shared_state = node.update_states(
                self.compute_events_statistics(
                    node.data_sample_keys,
                    shared_state=None,
                    _algo_name="Compute Events Statistics",
                ),
                local_state=None,
                round_idx=0,
                authorized_ids=set([node.organization_id]),
                aggregation_id=aggregation_node.organization_id,
                clean_models=False,
            )
            # keep the states in a list: one/organization
            shared_states.append(next_shared_state)

        aggregation_node.update_states(
            self.compute_agg_km_curve(
                shared_states=shared_states,
                _algo_name="Aggregate Events Statistics",
            ),
            round_idx=0,
            authorized_ids=set(
                train_data_node.organization_id for train_data_node in train_data_nodes
            ),
            clean_models=False,
        )

    @remote_data
    def compute_events_statistics(
        self,
        data_from_opener: pd.DataFrame,
        shared_state=None,
    ):
        """Compute events statistics for a subset of data.

        Parameters
        ----------
        datasamples : _type_
            _description_
        shared_state : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        del shared_state
        # we only use survival times
        propensity_cols = [
            col
            for col in data_from_opener.columns
            if col
            not in [
                self._duration_col,
                self._event_col,
                self._treated_col,
                self._client_identifier,
            ]
        ]

        X, y, treated, Xprop, _ = build_X_y_function(
            data_from_opener,
            self._event_col,
            self._duration_col,
            self._treated_col,
            self._target_cols,
            False,
            self._propensity_model,
            None,
            propensity_cols,
            self._tol,
            "iptw",
        )
        # X contains only the treatment column (strategy == iptw)

        X, _, weights = compute_X_y_and_propensity_weights_function(
            X, y, treated, Xprop, self._propensity_model, self._tol
        )

        # TODO actually use weights
        # del weights
        # retrieve times and events
        times = np.abs(y)
        events = y >= 0
        assert np.allclose(events, data_from_opener[self._event_col].values)
        treated = treated.astype(bool).flatten()

        return {
            "treated": compute_events_statistics(
                times[treated], events[treated], weights[treated]
            ),
            "untreated": compute_events_statistics(
                times[~treated], events[~treated], weights[~treated]
            ),
        }

    @remote
    def compute_agg_km_curve(self, shared_states):
        """Compute the aggregated Kaplan-Meier curve.

        Parameters
        ----------
        shared_states : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        treated_untreated_tnd_agg = {
            "treated": aggregate_events_statistics(
                [sh["treated"] for sh in shared_states]
            ),
            "untreated": aggregate_events_statistics(
                [sh["untreated"] for sh in shared_states]
            ),
        }

        return {
            "treated": km_curve(*treated_untreated_tnd_agg["treated"]),
            "untreated": km_curve(*treated_untreated_tnd_agg["untreated"]),
        }

    def save_local_state(self, path: Path):
        """Save the object on the disk.

        Should be used only by the backend, to define the local_state.

        Parameters
        ----------
        path : Path
            Where to save the object.
        """
        with open(path, "wb") as file:
            pk.dump(self.statistics_result, file)

    def load_local_state(self, path: Path) -> Any:
        """Load the object from the disk.

        Should be used only by the backend, to define the local_state.

        Parameters
        ----------
        path : Path
            Where to find the object.

        Returns
        -------
        Any
            Previously saved instance.
        """
        with open(path, "rb") as file:
            self.statistics_result = pk.load(file)
        return self
