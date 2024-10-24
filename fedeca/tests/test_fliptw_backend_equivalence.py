"""Tests for webdisco."""
import numpy as np
from pandas.testing import assert_frame_equal

from fedeca.fedeca_core import FedECA
from fedeca.tests.common import TestTempDir
from fedeca.utils.data_utils import generate_survival_data


# TODO increase rounds and an an assert to pooled equivalence as in
# aper simulations
class TestFLIPTWEnd2End(TestTempDir):
    """Webdisco tests class."""

    @classmethod
    def setUpClass(
        cls,
        n_clients=3,
        ndim=10,
        initial_step_size=0.95,
        seed=43,
        standardize_data=True,
        l1_ratio=0.0,
        penalizer=0.0,
        use_propensity=False,
        learning_rate_strategy="lifelines",
    ):
        """Initialize tests with data and FedIPTW object.

        Parameters
        ----------
        n_clients : int, optional
            The number of clients, by default 3
        ndim : int, optional
            The number of dimensions, by default 10
        initial_step_size : float, optional
            The first step size of NR descent, by default 0.95
        seed : int, optional
            The seed, by default 43
        standardize_data : bool, optional
            Whether or not to standardize data, by default True
        l1_ratio : float, optional
            The l1 ratio wrt L2., by default 0.0
        penalizer : float, optional
            The weight for the elasticnet penalty, by default 0.0
        learning_rate_strategy : str, optional
            How do we decrease the lr, by default "lifelines"
        """
        super().setUpClass()
        cls.n_clients = n_clients
        cls.df, _ = generate_survival_data(
            na_proportion=0.0,
            ncategorical=0,
            ndim=ndim,
            seed=seed,
            n_samples=1000,
            use_cate=False,
            censoring_factor=0.3,
        )
        # We can choose not to give any clients or data of any kind to FedECA
        # they will be given to it by the fit method
        cls.IPTWs = [
            FedECA(
                ndim=ndim,
                treated_col="treated",
                duration_col="T",
                event_col="E",
                num_rounds_list=[10, 10],
                initial_step_size=initial_step_size,
                seed=seed,
                standardize_data=standardize_data,
                l1_ratio=l1_ratio,
                penalizer=penalizer,
                learning_rate_strategy=learning_rate_strategy,
            )
            for _ in range(2)
        ]

    def test_fit(self):
        """Test end2end aplication of IPTW to synthetic data."""
        iptw_kwargs = {
            "data": self.df,
            "targets": None,
            "n_clients": self.n_clients,
            "split_method": "split_control_over_centers",
            "split_method_kwargs": {"treatment_info": "treated"},
            "data_path": self.test_dir,
            # "dp_target_epsilon": 2.,
            # "dp_max_grad_norm": 1.,
            # "dp_target_delta": 0.001,
            # "dp_propensity_model_training_params": {"batch_size": 100, "num_updates": 100},  # noqa: E501
            # "dp_propensity_model_optimizer_kwargs": {"lr": 0.01},
        }

        iptw_kwargs["backend_type"] = "subprocess"
        self.IPTWs[0].fit(**iptw_kwargs)
        iptw_kwargs["backend_type"] = "simu"
        self.IPTWs[1].fit(**iptw_kwargs)
        # TODO verify propensity model training wrt sklearn and full chain
        # vs iptw pooled implementation with sklearn and lifelines
        assert_frame_equal(self.IPTWs[0].results_, self.IPTWs[1].results_)
        assert np.allclose(self.IPTWs[0].lls[0], self.IPTWs[1].lls[0])

    @classmethod
    def tearDownClass(cls):
        """Tear down the class."""
        super(TestFLIPTWEnd2End, cls).tearDownClass()
        # We need to avoid persistence of DB in between TestCases, this is an obscure
        # hack but it's working
        first_client = cls.IPTWs[0].ds_client
        database = first_client._backend._db._db._data
        if len(database.keys()) > 1:
            for k in list(database.keys()):
                database.pop(k)
