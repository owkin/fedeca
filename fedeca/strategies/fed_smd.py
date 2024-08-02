from typing import List, Optional

import numpy as np
import pandas as pd
from substrafl import ComputePlanBuilder
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.nodes import AggregationNodeProtocol, TrainDataNodeProtocol
from substrafl.remote import remote, remote_data

from fedeca.utils.moments_utils import compute_global_moments, compute_uncentered_moment
from fedeca.utils.survival_utils import build_X_y_function


class FedSMD(ComputePlanBuilder):
    def __init__(
        self,
        duration_col,
        event_col,
        treated_col: str = None,
        propensity_model=None,
        tol: float = 1e-16,
    ):
        """FedSMD strategy. This class computes weighted SMD.

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
        self._propensity_model = propensity_model
        self._tol = tol

    def build_compute_plan(
        self,
        train_data_nodes: Optional[List[TrainDataNodeProtocol]],
        aggregation_node: Optional[List[AggregationNodeProtocol]],
        evaluation_strategy: Optional[EvaluationStrategy],
        num_rounds: Optional[int],
        clean_models: Optional[bool] = True,
    ):
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

        agg_shared_state = aggregation_node.update_states(
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
        del shared_state
        # we only use survival times
        cols = [
            col
            for col in datasamples.columns
            if col not in [self._duration_col, self._event_col, self._treated_col]
        ]
        X, _, treated, Xprop, _ = build_X_y_function(
            datasamples,
            self._event_col,
            self._duration_col,
            self._treated_col,
            self._target_cols,
            False,
            self._propensity_model,
            None,
            self._propensity_fit_cols,
            self._tol,
            "iptw",
        )
        raw_data = pd.DataFrame(X, columns=cols)
        weighted_data = pd.DataFrame(Xprop, columns=cols)
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
            results[res_name]["n_samples"] = (
                datasamples[mask_treatment].select_dtypes(include=np.number).count()
            )
        return results

    @remote
    def compute_smd(
        self,
        shared_states,
    ):
        """Computes Kaplan-Meier curve for a subset of data.

        Returns
        -------
        Placeholder
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

            smd_df = means_x.substract(means_y).div(
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
