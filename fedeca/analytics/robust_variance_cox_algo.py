"""Estimate the variance for mispecified Cox models."""
import copy
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from substrafl.remote import remote_data

from fedeca.algorithms import TorchWebDiscoAlgo
from fedeca.utils import make_substrafl_torch_dataset_class
from fedeca.utils.survival_utils import CoxPHModelTorch, compute_q_k


class RobustCoxVarianceAlgo(TorchWebDiscoAlgo):
    """Implement local client method for robust cox variance estimation."""

    def __init__(
        self,
        beta: np.ndarray,
        variance_matrix: np.ndarray,
        global_robust_statistics: dict[list[np.ndarray]],
        propensity_model: torch.nn.Module,
        duration_col: str,
        event_col: str,
        treated_col: str,
        standardize_data: bool = True,
        propensity_strategy: str = "iptw",
        dtype: float = "float64",
        tol: float = 1e-16,
    ):
        """Initialize Robust Cox Variance Algo.

        Parameters
        ----------
        beta : np.ndarray
            The weights of the trained Cox model.
        variance_matrix: np.ndarray
            The variance estimated in non robust mode aka H^{-1} rescaled
            by computed_stds.
        global_robust_statistics: dict[list[np.ndarray]]
            The global statistics on risk sets and events needed for FL
            computation.
        propensity_model: torch.nn.Module
            The propensity model trained.
        duration_col : str
            Column for the duration.
        event_col : str, optional
            Column for the event.
        treated_col : str, optional
            Column for the treatment.
        standardize_data : bool, optional
            Whether to standardize data. Defaults to True.
        propensity_strategy : str, optional
            Which covariates to use for the propensity model.
            Both give different results because of non-collapsibility:
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7986756/
            Defaults to iptw, which will use only the treatment allocation as covariate.
        dtype: str
            The type of the data to generate from dataframe. Defaults to float64.
        tol: float
            The clipping to avoid zero division errors.
        """
        self.beta = beta
        self.duration_col = duration_col
        self.treated_col = treated_col
        self.event_col = event_col
        self.standardize_data = standardize_data
        self.variance_matrix = variance_matrix
        self._tol = tol

        assert isinstance(global_robust_statistics, dict)
        global_robust_statistics_arg = copy.deepcopy(global_robust_statistics)

        assert all(
            [
                attr in global_robust_statistics_arg
                for attr in [
                    "global_weights_counts_on_events",
                    "global_risk_phi_list",
                    "global_risk_phi_x_list",
                    "distinct_event_times",
                    "global_moments",
                ]
            ]
        )

        global_moments = global_robust_statistics_arg.pop("global_moments")

        assert all(
            [
                len(global_robust_statistics_arg["global_weights_counts_on_events"])
                == len(v)
                for k, v in global_robust_statistics_arg.items()
                if k != "global_weights_counts_on_events"
            ]
        )

        self.global_robust_statistics = global_robust_statistics_arg
        if self.standardize_data:
            computed_stds = (
                global_moments["vars"]
                .transform(
                    lambda x: sqrt(x * global_moments["bias_correction"] + self._tol)
                )
                .to_numpy()
            )
        else:
            computed_stds = np.ones((self.variance_matrix.shape[0])).squeeze()

        # We initialize the Cox model to the final parameters from WebDisco
        # that we need to unnormalize
        fc1_weight = torch.from_numpy(beta * computed_stds)
        # We need to scale the variance matrix
        self.scaled_variance_matrix = (
            self.variance_matrix
            * np.tile(computed_stds, (self.variance_matrix.shape[0], 1)).T
        )

        class InitializedCoxPHModelTorch(CoxPHModelTorch):
            def __init__(self):
                super().__init__(ndim=1)
                self.fc1.weight.data = fc1_weight

        init_cox = InitializedCoxPHModelTorch()

        survival_dataset_class = make_substrafl_torch_dataset_class(
            [self.duration_col, self.event_col],
            self.event_col,
            self.duration_col,
            dtype=dtype,
        )
        super().__init__(
            model=init_cox,
            batch_size=sys.maxsize,
            dataset=survival_dataset_class,
            propensity_model=propensity_model,
            duration_col=duration_col,
            event_col=event_col,
            treated_col=treated_col,
            standardize_data=standardize_data,
            propensity_strategy=propensity_strategy,
            tol=tol,
        )
        # Now AND ONLY NOW we give it the global mean and weights computed by WebDisco
        # otherwise self.global_moments is set to None by
        # WebDisco init
        # TODO WebDisco init accept global_moments
        self.global_moments = global_moments

    @remote_data
    def local_q_computation(self, data_from_opener: pd.DataFrame, shared_state=None):
        """Compute Qk.

        Parameters
        ----------
        data_from_opener : pd.DataFrame
            Pandas dataframe provided by the opener.
        shared_state : None
            Unused here as this function only
            use local information already present in the data_from_opener.
            Defaults to None.

        Returns
        -------
        np.ndarray
            dictionary containing the local information on means, counts
            and number of sample. This dict will be used as a state to be
            shared to an AggregationNode in order to compute the aggregation
            of the different analytics.
        """
        df = data_from_opener

        distinct_event_times = self.global_robust_statistics["distinct_event_times"]
        weights_counts_on_events = self.global_robust_statistics[
            "global_weights_counts_on_events"
        ]
        risk_phi = self.global_robust_statistics["global_risk_phi_list"]
        risk_phi_x = self.global_robust_statistics["global_risk_phi_x_list"]

        (
            X_norm,
            y,
            weights,
        ) = self.compute_X_y_and_propensity_weights(df, shared_state=shared_state)

        self._model.eval()
        # The shape of expbetaTx is (N, 1)
        X_norm = torch.from_numpy(X_norm)
        score = self._model(X_norm).detach().numpy()
        X_norm = X_norm.numpy()

        phi_k, delta_betas_k, Qk = compute_q_k(
            X_norm,
            y,
            self.scaled_variance_matrix,
            distinct_event_times,
            weights_counts_on_events,
            risk_phi,
            risk_phi_x,
            score,
            weights,
        )

        # The attributes below are private to the client
        self._client_statistics = {}
        self._client_statistics["phi_k"] = phi_k
        self._client_statistics["delta_betas_k"] = delta_betas_k
        self._client_statistics["Qk"] = Qk

        return Qk

    def _get_state_to_save(self) -> dict:
        """Create the algo checkpoint: a dictionary saved with ``torch.save``.

        In this algo, it contains the state to save for every strategy.
        Reimplement in the child class to add strategy-specific variables.

        Example
        -------
        .. code-block:: python
            def _get_state_to_save(self) -> dict:
                local_state = super()._get_state_to_save()
                local_state.update({
                    "strategy_specific_variable": self._strategy_specific_variable,
                })
                return local_state
        Returns
        -------
        dict
            checkpoint to save
        """
        checkpoint = super()._get_state_to_save()
        checkpoint.update({"client_statistics": self._client_statistics})
        return checkpoint

    def _update_from_checkpoint(self, path: Path) -> dict:
        """Load the local state from the checkpoint.

        Parameters
        ----------
        path : pathlib.Path
            Path where the checkpoint is saved

        Returns
        -------
        dict
            Checkpoint
        """
        checkpoint = super()._update_from_checkpoint(path=path)
        self._client_statistics = checkpoint.pop("client_statistics")
        return checkpoint
