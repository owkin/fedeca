"""Module testing substraFL moments strategy."""
import os
import subprocess
from pathlib import Path

import git
import numpy as np
import torch
from lifelines import KaplanMeierFitter as KMF
from substrafl.dependency import Dependency
from substrafl.experiment import execute_experiment
from substrafl.model_loading import download_aggregate_shared_state
from substrafl.nodes import AggregationNode
from torch import nn

import fedeca
from fedeca.fedeca_core import LogisticRegressionTorch
from fedeca.strategies.fed_kaplan import FedKaplan
from fedeca.tests.common import TestTempDir
from fedeca.utils.data_utils import split_dataframe_across_clients
from fedeca.utils.survival_utils import CoxData


class TestKM(TestTempDir):
    """Test substrafl computation of KM.

    Tests the FL computation of KM is the same as in pandas-pooled version
    """

    def setUp(self, backend_type="subprocess", ndim=10) -> None:
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

    def test_end_to_end(self):
        """Compare a FL and pooled computation of Moments.

        The data are the tcga ones.
        """
        # Get fl_results.
        strategy = FedKaplan(
            treated_col="treatment",
            duration_col="time",
            event_col="event",
            propensity_model=self.propensity_model,
            client_identifier="center",
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

        X = self.df.drop(columns=["time", "event", "treatment"], axis=1)

        Xprop = torch.from_numpy(X.values).type(self.propensity_model.fc1.weight.dtype)
        with torch.no_grad():
            self.propensity_model.eval()
            propensity_scores = self.propensity_model(Xprop)

        propensity_scores = propensity_scores.detach().numpy().flatten()
        weights = self.df["treatment"] * 1.0 / propensity_scores + (
            1 - self.df["treatment"]
        ) * 1.0 / (1.0 - propensity_scores)
        weights = weights.values
        treatments = [1, 0]
        # TODO test with weights
        kms = [
            KMF().fit(
                durations=self.df.loc[self.df["treatment"] == t]["time"],
                event_observed=self.df.loc[self.df["treatment"] == t]["event"],
                weights=None,
            )
            for t in treatments
        ]
        s_gts = [kmf.survival_function_["KM_estimate"].to_numpy() for kmf in kms]
        grid_gts = [kmf.survival_function_.index.to_numpy() for kmf in kms]
        fl_grid_treated, fl_s_treated, _ = fl_results["treated"]
        fl_grid_untreated, fl_s_untreated, _ = fl_results["untreated"]

        assert np.allclose(fl_grid_treated, grid_gts[0], rtol=1e-2)
        assert np.allclose(fl_s_treated, s_gts[0], rtol=1e-2)
        assert np.allclose(fl_grid_untreated, grid_gts[1], rtol=1e-2)
        assert np.allclose(fl_s_untreated, s_gts[1], rtol=1e-2)
