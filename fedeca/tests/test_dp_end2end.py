"""Tests for DP training."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# from substrafl.model_loading import download_algo_state
from substrafl.strategies import FedAvg
from torch.optim import SGD

from fedeca.algorithms.torch_dp_fed_avg_algo import TorchDPFedAvgAlgo
from fedeca.fedeca_core import LogisticRegressionTorch
from fedeca.tests.common import TestTempDir
from fedeca.utils import (
    Experiment,
    make_accuracy_function,
    make_substrafl_torch_dataset_class,
)
from fedeca.utils.survival_utils import CoxData, make_categorical


# TODO increase rounds and an an assert to pooled equivalence as in
# aper simulations
class TestDPPropensityEnd2End(TestTempDir):
    """Webdisco tests class."""

    @classmethod
    def setUpClass(
        cls,
        n_clients=3,
        ndim=10,
        nsamples=300,
        seed=43,
    ):
        """Initialize tests with data and FedIPTW object.

        Parameters
        ----------
        n_clients : int, optional
            The number of clients, by default 3
        nsamples : int, optional
            The number of patients in total.
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
        rng = np.random.default_rng(seed)
        # Generating data with strong linear relationship
        simu_coxreg = CoxData(
            n_samples=nsamples,
            ndim=ndim,
            prop_treated=0.5,
            propensity="linear",
            dtype="float32",
            # Strong linearity
            overlap=100.0,
            seed=rng,
            random_censoring=True,
            censoring_factor=0.3,
            standardize_features=False,
        )
        X, T, C, treated, _ = simu_coxreg.generate_data()
        # Will make first columns to be categorical
        Xcat, Xcont = make_categorical(X, up_to=0)
        # Build the final dataframe using appropriate column names and adding
        # missing values
        cols_dict = {}
        X = np.concatenate((Xcat, Xcont), axis=1)
        for i in range(Xcat.shape[1] + Xcont.shape[1]):
            currentX = X[:, i].astype("float32")
            mask_na = rng.uniform(0, 1, X.shape[0]) > (1.0 - 0.0)
            currentX[mask_na] = np.nan
            if i < Xcat.shape[1]:
                colname = "cat_col"
            else:
                colname = "col"
                i -= Xcat.shape[1]
            cols_dict[f"{colname}_{i}"] = currentX

            #  The absolute value is superfluous but just to be sure
            cols_dict["T"] = np.abs(T)
            cols_dict["E"] = (1.0 - C).astype("uint8")
            cols_dict["treated"] = treated

            df = pd.DataFrame(cols_dict)
            # Final cast of categorical columns that was impossible due to nan in numpy
            for i in range(Xcat.shape[1]):
                df[f"cat_col_{i}"] = df[f"cat_col_{i}"].astype("Int64")

        cls.df = df
        accuracy_metrics_dict = {"accuracy": make_accuracy_function("treated")}
        logreg_dataset_class = make_substrafl_torch_dataset_class(
            ["treated"], "E", "T", dtype="float32", return_torch_tensors=True
        )
        cls.dp_trainings = []
        for i in range(2):
            num_rounds = 10
            logreg_model = LogisticRegressionTorch(ndim, torch.float32)
            optimizer = SGD(logreg_model.parameters(), lr=0.01)

            # Do not put self attributes in this class
            class DPLogRegAlgo(TorchDPFedAvgAlgo):
                def __init__(self):
                    super().__init__(
                        model=logreg_model,
                        criterion=nn.BCELoss(),
                        optimizer=optimizer,
                        dataset=logreg_dataset_class,
                        seed=seed,
                        num_updates=100,
                        batch_size=32,
                        num_rounds=num_rounds,
                        # industry standard
                        dp_target_epsilon=10.0,
                        # around 1/nsamples aroximately
                        dp_target_delta=0.001,
                        dp_max_grad_norm=1.0,
                    )

            dp_algo = DPLogRegAlgo()
            dp_fedavg_strategy = FedAvg(
                algo=dp_algo, metric_functions=accuracy_metrics_dict
            )
            dp_xp = Experiment(
                strategies=[dp_fedavg_strategy],
                num_rounds_list=[num_rounds],
            )
            cls.dp_trainings.append(dp_xp)

    def test_fit(self):
        """Test end2end aplication of DP FL to synthetic data."""
        dp_kwargs = {
            "data": self.df,
            "nb_clients": self.n_clients,
            "data_path": self.test_dir,
        }

        dp_kwargs["backend_type"] = "subprocess"
        self.dp_trainings[0].fit(**dp_kwargs)
        # final_algo = download_algo_state(
        #     client=self.dp_trainings[0].ds_client,
        #     compute_plan_key=self.dp_trainings[0].compute_plan_keys[0].key,
        #     round_idx=None,
        # )

        # final_model_subprocess = final_algo.model

        dp_kwargs["backend_type"] = "simu"
        self.dp_trainings[1].fit(**dp_kwargs)
        # final_model_simu = self.dp_trainings[1].train_data_nodes[0].algo.model

        # assert np.allclose(final_model_subprocess.fc1.weight.detach().numpy(),
        # final_model_simu.fc1.weight.detach().numpy())

    @classmethod
    def tearDownClass(cls):
        """Tear down the class."""
        super(TestDPPropensityEnd2End, cls).tearDownClass()
        # We need to avoid persistence of DB in between TestCases, this is an obscure
        # hack but it's working
        first_client = cls.dp_trainings[0].ds_client
        database = first_client._backend._db._db._data
        if len(database.keys()) > 1:
            for k in list(database.keys()):
                database.pop(k)
