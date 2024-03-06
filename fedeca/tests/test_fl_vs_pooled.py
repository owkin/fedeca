"""Tests for Robust IPTW."""
import pandas as pd
import pytest

from fedeca.competitors import PooledIPTW
from fedeca.fedeca_core import FedECA
from fedeca.tests.common import TestTempDir
from fedeca.utils.survival_utils import CoxData


class TestFedECAEnd2End(TestTempDir):
    """IPTW tests class."""

    # This is very long and thus should be only executed once that is why we use
    # setUpClass unlike setUp wich would otherwise get executed for each method
    @classmethod
    def setUpClass(
        cls,
        n_clients=3,
        ndim=100,
        nsamples=500,
        seed=43,
        variance_method="naive",
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
        """
        super().setUpClass()

        cls.n_clients = n_clients
        cls.nsamples = nsamples
        cls.seed = seed
        cls.variance_method = variance_method
        cls.ndim = ndim
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
        )
        cls.pooled_iptw.fit(cls.df)
        cls.pooled_iptw_results = cls.pooled_iptw.results_

        cls.fed_iptw = FedECA(
            ndim=cls.ndim,
            treated_col=cls._treated_col,
            duration_col=cls._duration_col,
            event_col=cls._event_col,
            num_rounds_list=[20, 20],
        )
        cls.fed_iptw.fit(
            data=cls.df,
            targets=None,
            n_clients=cls.n_clients,
            split_method="split_control_over_centers",
            split_method_kwargs={"treatment_info": cls._treated_col},
            backend_type="subprocess",
            variance_method=cls.variance_method,
            data_path=cls.test_dir,
        )
        cls.fed_iptw_results = cls.fed_iptw.results_

    @pytest.mark.slow
    def test_standard_deviations(self):
        """Test equality of end results.

        We allow ourselves rtol=1e-2 as in the paper.
        """
        pd.testing.assert_frame_equal(
            self.pooled_iptw_results.reset_index()[self.fed_iptw_results.columns],
            self.fed_iptw_results,
            rtol=1e-2,
        )


class TestRobustFedECAEnd2End(TestFedECAEnd2End):
    """RobustIPTW tests class."""

    @classmethod
    def setUpClass(cls):
        """Use parent class setup with robust=True."""
        super().setUpClass(variance_method="robust")


# TODO make this test pass and reactivate it
# class TestBtstFedECAEnd2End(TestFedECAEnd2End):
#     """BtstIPTW tests class."""

#     @classmethod
#     def setUpClass(cls):
#         """Use parent class setup with robust=True."""
#         super().setUpClass(variance_method="bootstrap")
