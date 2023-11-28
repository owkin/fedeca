"""Script to test robust variance."""
import lifelines
import numpy as np
from lifelines import CoxPHFitter

from fedeca.tests.common import TestTempDir
from fedeca.utils.survival_utils import (  # robust_sandwich_variance_pooled,
    CoxData,
    robust_sandwich_variance_distributed,
)


class TestRobustDistVarNumpy(TestTempDir):
    """Webdisco tests class."""

    @classmethod
    def setUpClass(
        cls,
        ndim=10,
        nsamples=1000,
        seed=1,
    ):
        """Initialize tests with data and FedIPTW object.

        Parameters
        ----------
        nsamples : int, optional
            The number of patients in total.
        ndim : int, optional
            The number of dimensions, by default 10
        seed : int, optional
            The seed, by default 43
        """
        super().setUpClass()
        cls.ndim = ndim
        cls.nsamples = nsamples
        cls.seed = seed
        rdm_state = np.random.default_rng(seed=cls.seed)
        # Generate data

        data = CoxData(
            seed=cls.seed,
            n_samples=cls.nsamples,
            ndim=cls.ndim,
            scale_t=10.0,
            shape_t=3.0,
            propensity="linear",
            standardize_features=False,
        )
        df = data.generate_dataframe()
        df = df.drop(columns=["propensity_scores", "treatment"], axis=1)
        df["weights"] = np.abs(rdm_state.normal(size=cls.nsamples))
        cls.df = df

        # Fit simple cox model with robust = False
        cls.lifelines_cph = CoxPHFitter()
        cls.lifelines_cph.fit(
            cls.df,
            duration_col="time",
            event_col="event",
            robust=False,
            weights_col="weights",
        )

        cls.non_robust_lifelines_variance = cls.lifelines_cph.variance_matrix_

        # Create labels as in FedECA setup
        cls.y = np.array(df["time"])
        cls.y[df["event"] == 0] = -1 * cls.y[df["event"] == 0]
        cls.X = df.drop(columns=["time", "event", "weights"], axis=1)

        # need to normalize as in lifelines
        cls.X = lifelines.utils.normalize(cls.X, cls.X.mean(0), cls.X.std(0))

        cls.lifelines_cph.fit(
            cls.df,
            duration_col="time",
            event_col="event",
            robust=True,
            weights_col="weights",
        )
        cls.true_variance = cls.lifelines_cph.variance_matrix_

        cls.beta = cls.lifelines_cph.params_ * cls.lifelines_cph._norm_std
        cls.weights = np.array(df["weights"].copy())
        cls.scaled_variance_matrix = (
            cls.non_robust_lifelines_variance
            * np.tile(cls.lifelines_cph._norm_std.values, (cls.ndim, 1)).T
        )

    def test_distributed_se_computation(self, n_clients=2):
        """Test equivalence with lifelines.

        Parameters
        ----------
        n_clients : int, optional
            The number of clients, by default 10
        """
        se = robust_sandwich_variance_distributed(
            np.array(self.X),
            self.y,
            self.beta,
            self.weights,
            self.scaled_variance_matrix,
            n_clients=n_clients,
        )

        np.testing.assert_allclose(
            se,
            self.lifelines_cph.summary["se(coef)"],
            rtol=1e-5,
        )
