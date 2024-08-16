"""Module testing substraFL moments strategy."""
import os
import subprocess
from pathlib import Path

import git
import numpy as np
import pandas as pd
import torch
from substrafl.dependency import Dependency
from substrafl.experiment import execute_experiment
from substrafl.model_loading import download_aggregate_shared_state
from substrafl.nodes import AggregationNode
from torch import nn

import fedeca
from fedeca.fedeca_core import LogisticRegressionTorch
from fedeca.metrics.metrics import standardized_mean_diff
from fedeca.strategies.fed_smd import FedSMD
from fedeca.tests.common import TestTempDir
from fedeca.utils.data_utils import split_dataframe_across_clients
from fedeca.utils.survival_utils import CoxData


class TestSMD(TestTempDir):
    """Test substrafl computation of SMD.

    Tests the FL computation of SMD is the same as in pandas-pooled version
    """

    def setUp(
        self, backend_type="subprocess", ndim=10, use_unweighted_variance=True
    ) -> None:
        """Set up the quantities needed for the tests."""
        # Let's generate 1000 data samples with 10 covariates
        data = CoxData(seed=42, n_samples=1000, ndim=ndim)
        self.df = data.generate_dataframe()

        # We remove the true propensity score
        self.df = self.df.drop(columns=["propensity_scores"], axis=1)

        self.clients, self.train_data_nodes, _, _, _ = split_dataframe_across_clients(
            self.df,
            n_clients=4,
            split_method="split_control_over_centers",
            split_method_kwargs={"treatment_info": "treatment"},
            data_path=Path(self.test_dir) / "data",
            backend_type=backend_type,
        )
        kwargs_agg_node = {"organization_id": self.train_data_nodes[0].organization_id}
        self.aggregation_node = AggregationNode(**kwargs_agg_node)
        # Packaging the right dependencies

        fedeca_path = fedeca.__path__[0]
        repo_folder = Path(
            git.Repo(fedeca_path, search_parent_directories=True).working_dir
        ).resolve()
        wheel_folder = repo_folder / "temp"
        os.makedirs(wheel_folder, exist_ok=True)
        for stale_wheel in wheel_folder.glob("fedeca*.whl"):
            stale_wheel.unlink()
        process = subprocess.Popen(
            f"python -m build --wheel --outdir {wheel_folder} {repo_folder}",
            shell=True,
            stdout=subprocess.PIPE,
        )
        process.wait()
        assert process.returncode == 0, "Failed to build the wheel"
        self.wheel_path = next(wheel_folder.glob("fedeca*.whl"))
        self.ds_client = self.clients[self.train_data_nodes[0].organization_id]
        self.propensity_model = LogisticRegressionTorch(ndim=ndim)

        self.propensity_model.fc1.weight.data = nn.parameter.Parameter(
            torch.randn(
                size=self.propensity_model.fc1.weight.data.shape, dtype=torch.float64
            )
        )
        self.propensity_model.fc1.bias.data = nn.parameter.Parameter(
            torch.randn(
                size=self.propensity_model.fc1.bias.data.shape, dtype=torch.float64
            )
        )
        self.use_unweighted_variance = use_unweighted_variance

    def test_end_to_end(self):
        """Compare a FL and pooled computation of Moments.

        The data are the tcga ones.
        """
        # Get fl_results.
        strategy = FedSMD(
            treated_col="treatment",
            duration_col="time",
            event_col="event",
            propensity_model=self.propensity_model,
            client_identifier="center",
            use_unweighted_variance=self.use_unweighted_variance,
        )

        compute_plan = execute_experiment(
            client=self.ds_client,
            strategy=strategy,
            train_data_nodes=self.train_data_nodes,
            evaluation_strategy=None,
            aggregation_node=self.aggregation_node,
            num_rounds=1,
            experiment_folder=str(Path(self.test_dir) / "experiment_summaries"),
            dependencies=Dependency(
                local_installable_dependencies=[Path(self.wheel_path)]
            ),
        )

        fl_results = download_aggregate_shared_state(
            client=self.ds_client,
            compute_plan_key=compute_plan.key,
            round_idx=0,
        )

        assert not fl_results["weighted_smd"].equals(fl_results["unweighted_smd"])
        X = self.df.drop(columns=["time", "event", "treatment"], axis=1)
        covariates = X.columns
        Xprop = torch.from_numpy(X.values).type(self.propensity_model.fc1.weight.dtype)
        with torch.no_grad():
            self.propensity_model.eval()
            propensity_scores = self.propensity_model(Xprop)

        propensity_scores = propensity_scores.detach().numpy().flatten()
        weights = self.df["treatment"] * 1.0 / propensity_scores + (
            1 - self.df["treatment"]
        ) * 1.0 / (1.0 - propensity_scores)
        weights = weights.values.flatten()

        X_df = pd.DataFrame(Xprop.numpy(), columns=covariates)

        standardized_mean_diff_pooled_weighted = standardized_mean_diff(
            X_df,
            self.df["treatment"] == 1,
            weights=weights,
            use_unweighted_variance=self.use_unweighted_variance,
        ).div(100.0)
        standardized_mean_diff_pooled_unweighted = standardized_mean_diff(
            X_df,
            self.df["treatment"] == 1,
            use_unweighted_variance=self.use_unweighted_variance,
        ).div(100.0)

        # We check equality of FL computation and pooled results
        pd.testing.assert_series_equal(
            standardized_mean_diff_pooled_unweighted,
            fl_results["unweighted_smd"],
            rtol=1e-2,
        )
        pd.testing.assert_series_equal(
            standardized_mean_diff_pooled_weighted,
            fl_results["weighted_smd"],
            rtol=1e-2,
        )


class TestSMDWeightedVar(TestSMD):
    def setUp(self) -> None:
        super().setUp(use_unweighted_variance=False)
