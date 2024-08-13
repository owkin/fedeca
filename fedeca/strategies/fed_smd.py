"""Compute SMD for weighted and unweighted data in FL."""
import pickle as pk
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
from substrafl import ComputePlanBuilder
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.nodes import AggregationNodeProtocol, TrainDataNodeProtocol
from substrafl.remote import remote, remote_data

from fedeca.utils.moments_utils import compute_global_moments, compute_uncentered_moment
from fedeca.utils.survival_utils import (
    build_X_y_function,
    compute_X_y_and_propensity_weights_function,
)


class FedSMD(ComputePlanBuilder):
    """Compute SMD for weighted and unweighted data in FL.

    Parameters
    ----------
    ComputePlanBuilder : ComputePlanBuilder
        Analytics strategy.
    """

    def __init__(
        self,
        duration_col: str,
        event_col: str,
        treated_col: str,
        propensity_model: torch.nn.Module,
        client_identifier: str,
        tol: float = 1e-16,
    ):
        """Initialize FedSMD strategy.

        This class computes weighted SMD.

        Parameters
        ----------
        treated_col : Union[None, str], optional
            The column describing the treatment, by default None
        duration_col : str
            The column describing the duration of the event, by default None

        propensity_model : Union[None, nn.Module], optional
            _description_, by default None
        """
        super().__init__()

        self._duration_col = duration_col
        self._event_col = event_col
        self._treated_col = treated_col
        self._target_cols = [self._duration_col, self._event_col]
        self._propensity_model = propensity_model
        self._propensity_fit_cols = None
        self._client_identifier = client_identifier
        self._tol = tol
        self.statistics_result = None

        # Populating kwargs for reinstatiation
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
                self.compute_local_moments_per_group(
                    node.data_sample_keys,
                    shared_state=None,
                    _algo_name="Compute local moments per group",
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
            self.compute_smd(
                shared_states=shared_states,
                _algo_name="compute smd for weighted and unweighted data",
            ),
            round_idx=0,
            authorized_ids=set(
                train_data_node.organization_id for train_data_node in train_data_nodes
            ),
            clean_models=False,
        )

    @remote_data
    def compute_local_moments_per_group(
        self,
        data_from_opener,
        shared_state=None,
    ):
        """Compute events statistics for a subset of data.

        Parameters
        ----------
        data_from_opener: pd.DataFrame
            Data to compute statistics on.
        shared_state: list[pd.DataFrame]
            List of shared states.

        Returns
        -------
        dict
            Method output or placeholder thereof.
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

        # X contains only the treatment column (strategy == iptw)
        # we use Xprop which contain all propensity columns, which
        # are the only ones we are interested in
        raw_data = pd.DataFrame(Xprop, columns=propensity_cols)
        weighted_data = pd.DataFrame(
            np.multiply(weights, Xprop), columns=propensity_cols
        )
        results = {}
        for treatment in [0, 1]:
            mask_treatment = treated == treatment
            res_name = "treated" if treatment else "untreated"
            results[res_name] = {}
            results[res_name]["weighted"] = {
                f"moment{k}": compute_uncentered_moment(
                    weighted_data[mask_treatment], k
                )
                for k in range(1, 3)
            }
            results[res_name]["unweighted"] = {
                f"moment{k}": compute_uncentered_moment(raw_data[mask_treatment], k)
                for k in range(1, 3)
            }
            results[res_name]["unweighted"]["n_samples"] = results[res_name][
                "weighted"
            ]["n_samples"] = (
                raw_data[mask_treatment].select_dtypes(include=np.number).count()
            )
        return results

    @remote
    def compute_smd(
        self,
        shared_states,
    ):
        """Compute Kaplan-Meier curve for a subset of data.

        Parameters
        ----------
        shared_states: list
            List of shared states.

        Returns
        -------
        dict
            Method output or placeholder thereof.
        """

        def std_mean_differences(x, y):
            """Compute standardized mean differences."""
            means_x = x["global_uncentered_moment_1"]
            # we match nump std with 0 ddof contrary to standarization for Cox
            stds_x = np.sqrt(x["global_centered_moment_2"] + self._tol)

            means_y = y["global_uncentered_moment_1"]
            # we match nump std with 0 ddof contrary to standarization for Cox
            stds_y = np.sqrt(y["global_centered_moment_2"] + self._tol)

            smd_df = means_x.subtract(means_y).div(
                stds_x.pow(2).add(stds_y.pow(2)).div(2).pow(0.5)
            )
            return smd_df

        treated_raw = compute_global_moments(
            [shared_state["treated"]["unweighted"] for shared_state in shared_states]
        )
        untreated_raw = compute_global_moments(
            [shared_state["untreated"]["unweighted"] for shared_state in shared_states]
        )

        smd_raw = std_mean_differences(treated_raw, untreated_raw)

        treated_weighted = compute_global_moments(
            [shared_state["treated"]["weighted"] for shared_state in shared_states]
        )
        untreated_weighted = compute_global_moments(
            [shared_state["untreated"]["weighted"] for shared_state in shared_states]
        )

        smd_weighted = std_mean_differences(treated_weighted, untreated_weighted)

        return {"weighted_smd": smd_weighted, "unweighted_smd": smd_raw}

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
