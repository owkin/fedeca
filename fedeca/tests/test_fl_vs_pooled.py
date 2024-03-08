"""Tests for Robust IPTW."""
import copy

import numpy as np
import pandas as pd
import pytest

from fedeca.competitors import PooledIPTW
from fedeca.fedeca_core import FedECA
from fedeca.tests.common import TestTempDir
from fedeca.utils.data_utils import split_control_over_centers
from fedeca.utils.survival_utils import CoxData


class TestFedECAEnd2End(TestTempDir):
    """IPTW tests class."""

    # This is very long and thus should be only executed once that is why we use
    # setUpClass unlike setUp wich would otherwise get executed for each method
    @classmethod
    def setUpClass(
        cls,
        n_clients=3,
        ndim=10,
        nsamples=500,
        seed=42,
        variance_method="naive",
        bootstrap_seeds=None,
        bootstrap_function=None,
    ):
        """Set up the test class for experiment comparison.

        Parameters
        ----------
        n_clients : int
            The number of clients in the federation
        nsamples : int
            The number of samles in total.
        seed : int
            The seed to use for the data generation process.
        variance_method : str
            The variance method to use for the IPTW.
        bootstrap_seeds : list[Seeds]
            The seeds to use for the bootstrap variance method.
        bootstrap_function: Union[Callable, None]
            The bootstrap function to use to mimic the global sampling.
        """
        super().setUpClass()

        cls.n_clients = n_clients
        cls.nsamples = nsamples
        cls.seed = seed
        cls.variance_method = variance_method
        cls.ndim = ndim
        cls.bootstrap_seeds = bootstrap_seeds
        cls.n_bootstrap = len(bootstrap_seeds)
        data = CoxData(seed=cls.seed, n_samples=cls.nsamples, ndim=cls.ndim)
        df = data.generate_dataframe()
        cls.df = df.drop(columns=["propensity_scores"], axis=1)
        cls._treated_col = "treatment"
        cls._event_col = "event"
        cls._duration_col = "time"

        cls.pooled_iptw = PooledIPTW(
            treated_col=cls._treated_col,
            event_col=cls._event_col,
            duration_col=cls._duration_col,
            variance_method=cls.variance_method,
            seed=cls.seed,
            n_bootstrap=cls.n_bootstrap,
        )
        cls.pooled_iptw.fit(cls.df)
        cls.pooled_iptw_results = cls.pooled_iptw.results_

        cls.fed_iptw = FedECA(
            ndim=cls.ndim,
            treated_col=cls._treated_col,
            duration_col=cls._duration_col,
            event_col=cls._event_col,
            num_rounds_list=[50, 50],
        )

        if variance_method == "bootstrap":
            # We need for the client to know who they are so that we can finely
            # control the per-client sampling
            clients_indices = split_control_over_centers(
                cls.df,
                n_clients=cls.n_clients,
                treatment_info=cls._treated_col,
                seed=42,
            )
            cls.df["client"] = -1
            for idx, client_indices in enumerate(clients_indices):
                cls.df["client"].iloc[client_indices] = f"client_{idx}"
            # the following is neede because 1 we need original indices for
            # correct broadcasting and 2 we need to drop both client and
            # original indices at inference time where data si not bootstraped
            # We cannot just stringify the indices as the dump on disk would make
            # pandas consider them as integers instead of string that is why we
            # add the _str suffix
            cls.df["original_clients_indices"] = pd.Series(
                [f"{e}_str" for e in range(len(cls.df.index))], dtype="string"
            )

        cls.fed_iptw.fit(
            data=cls.df,
            targets=None,
            n_clients=cls.n_clients,
            split_method="split_control_over_centers",
            split_method_kwargs={"treatment_info": cls._treated_col},
            backend_type="simu",
            variance_method=cls.variance_method,
            data_path=cls.test_dir,
            bootstrap_seeds=cls.bootstrap_seeds,
            bootstrap_function=bootstrap_function,
        )
        cls.fed_iptw_results = cls.fed_iptw.results_

    @pytest.mark.slow
    def test_matching(self, rtol=1e-2):
        """Test equality of end results.

        We allow ourselves rtol=1e-2 as in the paper.
        """
        pd.testing.assert_frame_equal(
            self.pooled_iptw_results.reset_index()[self.fed_iptw_results.columns],
            self.fed_iptw_results,
            rtol=rtol,
        )


class TestRobustFedECAEnd2End(TestFedECAEnd2End):
    """RobustIPTW tests class."""

    @classmethod
    def setUpClass(cls):
        """Use parent class setup with robust=True."""
        super().setUpClass(variance_method="robust")


class TestBtstFedECAEnd2End(TestFedECAEnd2End):
    """BtstIPTW tests class."""

    @classmethod
    def setUpClass(cls, seed=42, n_bootstrap=2):
        """Use parent class setup with robust=True."""
        bootstrap_seeds = []
        cls.n_bootstrap = n_bootstrap
        # First one called in init of the class of PooledIPTW
        cls.seed = np.random.default_rng(seed)
        # Second one called when calling bootstrap_std
        cls.seed = np.random.default_rng(cls.seed)
        data = CoxData(seed=42, n_samples=500, ndim=10)
        df = data.generate_dataframe()
        df = df.drop(columns=["propensity_scores"], axis=1)

        # Tracing random state AND indices
        clients_indices = split_control_over_centers(
            df, n_clients=3, treatment_info="treatment", seed=42
        )
        bootstrap_seeds = []
        clients_btst_samples = []
        for _ in range(cls.n_bootstrap):
            rng = np.random.default_rng(cls.seed)
            rng_copy = copy.deepcopy(rng)
            bootstrap_seeds.append(rng_copy)
            new_df = df.sample(df.shape[0], replace=True, random_state=rng)
            current_sampled_points = {}
            for idx, client_indices in enumerate(clients_indices):
                # One cannot use sets as there are duplicates in the indices
                current_sampled_points[f"client_{idx}"] = [
                    idx for idx in new_df.index if idx in client_indices
                ]

            clients_btst_samples.append(current_sampled_points)

        def bootstrap_function(data, seed):
            data = copy.deepcopy(data)
            if not isinstance(seed, int):
                rng = copy.deepcopy(seed)
                bootstrap_id = [b.__getstate__() for b in bootstrap_seeds].index(
                    rng.__getstate__()
                )
            else:
                rng = seed
                bootstrap_id = bootstrap_seeds.index(rng)

            assert data["client"].nunique() == 1
            client_id = data["client"].iloc[0]
            data = data.drop(columns=["client"])
            sampled_rows = clients_btst_samples[bootstrap_id][client_id]
            # We convert it back to int
            data["original_clients_indices"] = [
                int(e.split("_")[0]) for e in data["original_clients_indices"].tolist()
            ]
            # We assert this is indeed a sampling of rows of this client
            assert all(
                [
                    s_row in data["original_clients_indices"].tolist()
                    for s_row in sampled_rows
                ]
            )
            data = (
                data.set_index("original_clients_indices")
                .loc[sampled_rows]
                .reset_index()
            )
            assert all(
                [
                    a == b
                    for a, b in zip(
                        data["original_clients_indices"].tolist(), sampled_rows
                    )
                ]
            )
            data = data.drop(columns={"original_clients_indices"})
            return data

        super().setUpClass(
            variance_method="bootstrap",
            bootstrap_seeds=bootstrap_seeds,
            seed=42,
            bootstrap_function=bootstrap_function,
        )

    def test_matching(self):
        """Changing tolerance as we can't match precise seeds in bootstrap."""
        super().test_matching(rtol=0.1)
