"""Tests for webdisco."""
import copy
import sys
from math import sqrt

import lifelines
import numpy as np
import pandas as pd
import pytest
import torch
from autograd import elementwise_grad
from autograd import numpy as anp
from lifelines.utils import StepSizer, concordance_index
from scipy.linalg import inv, norm
from scipy.linalg import solve as spsolve
from substrafl.algorithms.pytorch import weight_manager
from torch import nn

from fedeca.algorithms import TorchWebDiscoAlgo
from fedeca.strategies import WebDisco
from fedeca.strategies.webdisco_utils import (
    compute_summary_function,
    get_final_cox_model_function,
)
from fedeca.tests.common import TestTempDir
from fedeca.utils.data_utils import generate_cox_data_and_substra_clients
from fedeca.utils.substrafl_utils import (
    Experiment,
    get_outmodel_function,
    make_c_index_function,
    make_substrafl_torch_dataset_class,
)
from fedeca.utils.survival_utils import (
    CoxPHModelTorch,
    analytical_gradient_cox_partial_loglikelihood_breslow_torch,
    cox_partial_loglikelihood_breslow_torch,
    hessian_torch,
)

DTYPES_TORCH = {"float32": torch.float, "float64": torch.double}


class TestWebDisco(TestTempDir):
    """Webdisco tests class."""

    @classmethod
    def tearDownClass(cls):
        """Tear down the class."""
        super(TestWebDisco, cls).tearDownClass()
        # We need to avoid persistence of DB in between TestCases, this is an obscure
        # hack but it's working
        first_client = cls.clients[list(cls.clients.keys())[0]]
        database = first_client._backend._db._db._data
        if len(database.keys()) > 1:
            for k in list(database.keys()):
                database.pop(k)

    @classmethod
    def get_lifelines_results(cls):
        """Get lifelines results."""
        # Fitting pooled data with lifelines and check
        # that lifelines give decent results
        cls.cphf = lifelines.fitters.coxph_fitter.CoxPHFitter(
            penalizer=cls.penalizer, l1_ratio=cls.l1_ratio
        )
        lifelines_kwargs = {
            "duration_col": cls._duration_col,
            "event_col": cls._event_col,
            "robust": cls.robust,
        }
        if cls.use_propensity:
            lifelines_kwargs["weights_col"] = "weights"

        cls.cphf.fit(cls.df, **lifelines_kwargs)

        # Removing lifelines specific preprocessing
        if "weights" in cls.df:
            cls.df = cls.df.drop(columns=["weights"])

        # If we used propensity then I guess coefficients might differ
        if not cls.use_propensity and np.allclose(cls.penalizer, 0.0):
            # To get closer to the true Cox coefficients one has to reduce
            # censorship + add more data points. With 100000 data points and no
            # censorship one can get pretty good estimates (but still not
            # perfect) approx rtol=1e-2
            assert (
                np.linalg.norm(cls.coeffs - np.array(cls.cphf.params_)) < 1.0
            ), "Lifelines could not fit the data."

    # This is very long and thus should be only executed once that is why we use
    # setUpClass unlike setUp wich would otherwise get executed for each method
    @classmethod
    def setUpClass(
        cls,
        backend="pytorch",
        n_clients=3,
        n_samples_per_client=100,
        ndim=10,
        dtype="float64",
        initial_step_size=0.95,
        seed=43,
        standardize_data=True,
        l1_ratio=0.0,
        penalizer=0.0,
        use_propensity=False,
        learning_rate_strategy="lifelines",
        robust=False,
        run=True,
    ):
        """Set up the test class for experiment comparison.

        Parameters
        ----------
        cls : TestCase class
            Test class instance.
        backend : str, optional
            Backend type, by default "pytorch".
        n_clients : int, optional
            Number of clients, by default 3.
        n_samples_per_client : int, optional
            Number of samples per client, by default 100.
        ndim : int, optional
            Number of dimensions, by default 10.
        dtype : str, optional
            Data type, by default "float64".
        initial_step_size : float, optional
            Initial step size, by default 0.95.
        seed : int, optional
            Random seed, by default 43.
        standardize_data : bool, optional
            Standardize data, by default True.
        l1_ratio : float, optional
            L1 ratio, by default 0.0.
        penalizer : float, optional
            Penalizer, by default 0.0.
        use_propensity : bool, optional
            Use propensity scores, by default False.
        learning_rate_strategy : str, optional
            Learning rate strategy, by default "lifelines".
        robust: bool, optional
            Whether to perform robust variance estimation.
        run : bool, optional
            Whether to run WebDisco or not.
        """
        super().setUpClass()
        cls.n_clients = n_clients
        cls.backend = backend
        # Data creation, we use appropriate data to avoid
        # ill conditionned hessian
        cls.ndim = ndim
        cls.dtype = dtype
        cls.penalizer = penalizer
        cls.l1_ratio = l1_ratio
        cls.standardize_data = standardize_data
        cls.initial_step_size = initial_step_size
        cls.seed = seed
        cls.use_propensity = use_propensity
        cls.learning_rate_strategy = learning_rate_strategy
        cls.robust = robust

        # Creating pooled data from parameters
        # Note that this could probably be factorized across TestCases
        (
            cls.clients,
            cls.train_data_nodes,
            cls.test_data_nodes,
            cls.dfs,
            cls.df,
            cls.coeffs,
        ) = generate_cox_data_and_substra_clients(
            n_clients=cls.n_clients,
            ndim=cls.ndim,
            backend_type="subprocess",
            data_path=cls.test_dir,
            seed=seed,
            n_per_client=n_samples_per_client,
            add_treated=cls.use_propensity,
        )

        assert cls.coeffs.shape == (cls.ndim,)

        cls.ds_client = cls.clients[list(cls.clients)[0]]

        assert len(cls.df.index) == len(
            cls.df["T"].unique()
        ), "There are ties, lifelines and webdisco will differ"

        cls._target_cols = ["T", "E"]
        cls._duration_col = "T"
        cls._event_col = "E"
        if cls.use_propensity:
            cls._treated_col = "treated"
        else:
            cls._treated_col = None

        # We order the dataframe to use lifelines get_efron function
        sort_by = [cls._duration_col, cls._event_col]
        cls.df = cls.df.sort_values(by=sort_by)

        # We compute lots of reference quantities in "pooled" setting
        cls.X = cls.df.drop(columns=cls._target_cols).to_numpy(cls.dtype)
        cls.standardize_data = standardize_data
        if cls.standardize_data:
            cls.true_means = cls.X.mean(axis=0)
            # Very important to match pandas
            cls.true_stds = cls.X.std(axis=0, ddof=1)
            cls.X -= cls.true_means
            cls.X /= cls.true_stds

        if cls.use_propensity:
            propensity_strategy = "aiptw"

            class LogisticRegressionTorch(nn.Module):
                def __init__(self):
                    super(LogisticRegressionTorch, self).__init__()
                    self.fc1 = nn.Linear(cls.ndim, 1).to(torch.float64)

                def forward(self, x, eval=False):
                    x = self.fc1(x)
                    return torch.sigmoid(x)

            torch.manual_seed(cls.seed)
            propensity_model = LogisticRegressionTorch()
            propensity_model.eval()
            # Xprop is neither standardized nor contains treated
            cls.Xprop = cls.df.drop(
                columns=cls._target_cols + [cls._treated_col]
            ).to_numpy(cls.dtype)
            with torch.no_grad():
                propensity_scores = (
                    propensity_model(torch.from_numpy(cls.Xprop)).detach().numpy()
                )
            treated = cls.df[cls._treated_col].to_numpy().reshape((-1, 1))
            cls.weights = treated * 1.0 / propensity_scores + (1 - treated) * 1.0 / (
                1.0 - propensity_scores
            )
            # This is only for lifelines then we need to remove it
            cls.df["weights"] = cls.weights
        else:
            propensity_model = None
            cls.weights = None
            propensity_strategy = "aiptw"  # None is not supported

        cls.E = cls.df["E"].to_numpy(cls.dtype)
        cls.df["time_multiplier"] = [2.0 * e - 1.0 for e in cls.df["E"].tolist()]
        cls.t = (cls.df["T"] * cls.df["time_multiplier"]).to_numpy(cls.dtype)
        cls.df = cls.df.drop(columns=["time_multiplier"])

        cls.get_lifelines_results()

        # A round is defined by a local training step followed by an aggregation
        # operation
        # This was hand-tuned don't change
        cls.NUM_ROUNDS = 8
        torch_dtype = DTYPES_TORCH[cls.dtype]
        if cls.use_propensity:
            if propensity_strategy == "aiptw":
                ndim = cls.X.shape[1]
            else:
                ndim = 1
        else:
            ndim = cls.X.shape[1]

        cls.model = CoxPHModelTorch(ndim=ndim, torch_dtype=torch_dtype)

        cls.dataset = make_substrafl_torch_dataset_class(
            cls._target_cols, cls._event_col, cls._duration_col, dtype=cls.dtype
        )

        # Needed for MyAlgo to avoid confusing the cls of this class and of MyAlgo
        # (bug in substrafl)
        model = cls.model
        dataset = cls.dataset
        duration_col = cls._duration_col
        event_col = cls._event_col
        treated_col = cls._treated_col

        class MyAlgo(TorchWebDiscoAlgo):
            def __init__(self, *args, **kwargs):
                del args
                del kwargs
                super().__init__(
                    model=model,
                    batch_size=sys.maxsize,
                    dataset=dataset,
                    seed=seed,
                    duration_col=duration_col,
                    event_col=event_col,
                    treated_col=treated_col,
                    standardize_data=standardize_data,
                    penalizer=penalizer,
                    l1_ratio=l1_ratio,
                    initial_step_size=initial_step_size,
                    learning_rate_strategy=learning_rate_strategy,
                    propensity_model=propensity_model,
                    propensity_strategy=propensity_strategy,
                    store_hessian=True,
                )

        cls.strategy = WebDisco(algo=MyAlgo(), standardize_data=cls.standardize_data)
        cls.cindex = make_c_index_function(cls._duration_col, cls._event_col)
        cls.webdisco_experiment = Experiment(
            ds_client=cls.clients[list(cls.clients.keys())[0]],
            strategies=[cls.strategy],
            train_data_nodes=cls.train_data_nodes,
            num_rounds_list=[cls.NUM_ROUNDS],
            metrics_dicts_list=[{"C-index": cls.cindex}],
            experiment_folder=cls.test_dir,
        )
        if cls.run:
            cls.webdisco_experiment.run()

    def test_aggregate_statistics(self):
        """Test the aggregated statistics."""
        if not self.use_propensity:
            # We retrieve the global survival statistics computed by WebDisco
            global_survival_statistics = get_outmodel_function(
                "Compute global statistics from local quantities",
                client=self.ds_client,
                compute_plan_key=self.webdisco_experiment.compute_plan_keys[0].key,
            )["global_survival_statistics"]
            computed_distinct_event_times = global_survival_statistics[
                "distinct_event_times"
            ]

            computed_list_nds = global_survival_statistics["list_number_events_by_time"]
            computed_statistics = global_survival_statistics[
                "global_sum_features_on_events"
            ]
            computed_number_of_distinct_values = global_survival_statistics[
                "num_global_events_time"
            ]
            # True statistics and distinct values
            true_distinct_event_times = np.unique(self.t[self.t > 0])
            true_n_distinct_event_times = len(true_distinct_event_times)
            true_statistics = np.zeros_like(self.X[0])
            true_nds = []
            for v in true_distinct_event_times:
                indices = np.where(self.t == v)[0]
                true_statistics += self.X[indices].sum(axis=0)
                true_nds.append(len(indices))
            assert np.allclose(true_distinct_event_times, computed_distinct_event_times)
            assert np.allclose(
                true_n_distinct_event_times, computed_number_of_distinct_values
            )
            assert np.allclose(true_nds, [sum(e) for e in zip(*computed_list_nds)])
            assert np.allclose(
                true_statistics, computed_statistics
            ), computed_statistics

    def test_compute_true_moments(self):
        """Test the computation of the moments."""
        if self.standardize_data:
            # We retrieve the global moments computed by WebDisco
            aggregated_moments = get_outmodel_function(
                "Compute the global centered moments given the local results.",
                client=self.ds_client,
                compute_plan_key=self.webdisco_experiment.compute_plan_keys[0].key,
            )
            computed_means = aggregated_moments["global_uncentered_moment_1"].to_numpy()
            computed_vars = aggregated_moments["global_centered_moment_2"]
            bias_correction = (aggregated_moments["total_n_samples"]) / float(
                aggregated_moments["total_n_samples"] - 1
            )
            computed_stds = computed_vars.transform(
                lambda x: sqrt(x * bias_correction + 1e-16)
            )

            assert np.allclose(computed_means, self.true_means)
            assert np.allclose(computed_stds, self.true_stds)

    def test_newton_raphson(self):
        """Test newton raphson algorithm."""
        # We use the initial model as a starting point
        coxmodel = copy.deepcopy(self.model)
        n = self.X.shape[0]

        # We do batch Gradient Newton-Raphson with lifelines tricks and compare it back
        # to WebDisco
        stepsizer = StepSizer(self.initial_step_size)
        for i in range(self.NUM_ROUNDS):
            coxmodel.zero_grad()
            coxpl = cox_partial_loglikelihood_breslow_torch(coxmodel, self.X, self.t)
            coxpl.backward()
            # We compute the analytical gradient in the pooled case
            ana_grad = analytical_gradient_cox_partial_loglikelihood_breslow_torch(
                coxmodel, self.X, self.t
            )
            # We compare it to torch autodiff just to be sure
            assert torch.allclose(coxmodel.fc1.weight.grad, ana_grad)
            true_gradient = coxmodel.fc1.weight.grad
            # We compute the hessian
            true_hessian = hessian_torch(
                cox_partial_loglikelihood_breslow_torch(coxmodel, self.X, self.t),
                coxmodel.fc1.weight,
            ).squeeze()
            # Check hessian and gradients
            webdisco_gradient_and_hessian = get_outmodel_function(
                "Compute gradient and hessian",
                client=self.ds_client,
                compute_plan_key=self.webdisco_experiment.compute_plan_keys[0].key,
                idx_task=i,
            )
            if self.penalizer > 0.0:
                webdisco_gradient_and_hessian_client = get_outmodel_function(
                    "Training with MyAlgo",
                    client=self.ds_client,
                    compute_plan_key=self.webdisco_experiment.compute_plan_keys[0].key,
                    idx_task=i * self.n_clients,
                )
            webdisco_gradient = torch.from_numpy(
                webdisco_gradient_and_hessian["gradient"]
            )
            webdisco_hessian = torch.from_numpy(
                webdisco_gradient_and_hessian["hessian"]
            )

            # We test the resulting hessian and gradients
            if not self.use_propensity:
                # We test against "true" gradient when it's not weighted
                assert torch.allclose(true_gradient, -webdisco_gradient, atol=1e-4)
                assert torch.allclose(
                    true_hessian.squeeze(), -webdisco_hessian, atol=1e-4
                )

            class FakeDF:
                def __init__(self, values):
                    self.values = values

            # We always test against lifelines
            if self.weights is None:
                self.weights = np.ones((self.X.shape[0],))

            (
                lifelines_hessian,
                lifelines_gradient,
                _,
            ) = self.cphf._get_efron_values_single(
                FakeDF(self.X),
                self.df[self._duration_col],
                self.df[self._event_col],
                pd.Series(self.weights.squeeze()),
                entries=None,
                beta=coxmodel.fc1.weight.data.detach().numpy().squeeze(),
            )

            if self.penalizer > 0.0:
                # We use a smooth approximation for the L1 norm (for more details
                # see docstring of function)
                # we use numpy autograd to be able to compute the first and second
                # order derivatives of this expression
                current_weights = coxmodel.fc1.weight.data.detach().numpy()
                alpha = 1.3 ** (i + 1)

                def soft_abs(x, a):
                    return 1 / a * (anp.logaddexp(0, -a * x) + anp.logaddexp(0, a * x))

                def elastic_net_penalty(beta, a):
                    l1 = self.l1_ratio * soft_abs(beta, a)
                    l2 = 0.5 * (1 - self.l1_ratio) * (beta**2)
                    reg = n * (self.penalizer * (l1 + l2)).sum()
                    return reg

                # Regularization affects both the gradient and the hessian
                # producing a better conditioned hessian.
                d_elastic_net_penalty = elementwise_grad(elastic_net_penalty)
                dd_elastic_net_penalty = elementwise_grad(d_elastic_net_penalty)
                # lifelines trick to progressively sharpen the approximation of
                # the l1 regularization.
                # We are trying to **maximize** the log-likelihood that is why
                # we put a negative sign and not a plus sign on the regularization.
                # The fact that we are actually moving towards the maximum and
                # not towards the minimum is because -H is psd.
                true_gradient += d_elastic_net_penalty(current_weights, alpha)
                true_hessian[
                    np.diag_indices(max(true_hessian.shape))
                ] += dd_elastic_net_penalty(current_weights, alpha)
                lifelines_gradient -= d_elastic_net_penalty(
                    current_weights, alpha
                ).squeeze()
                lifelines_hessian[
                    np.diag_indices(max(lifelines_hessian.shape))
                ] -= dd_elastic_net_penalty(current_weights, alpha).squeeze()
                # WebDisco does that internally but it only appears in the client's side
                webdisco_hessian = webdisco_gradient_and_hessian_client["server_state"][
                    "hessian"
                ]
                webdisco_gradient = webdisco_gradient_and_hessian_client[
                    "server_state"
                ]["gradient"]

            if not self.use_propensity:
                assert np.allclose(webdisco_hessian, -true_hessian, atol=1e-4)
                assert np.allclose(webdisco_gradient, -true_gradient, atol=1e-4)
            assert np.allclose(webdisco_hessian, lifelines_hessian, atol=1e-4)
            assert np.allclose(webdisco_gradient, lifelines_gradient, atol=1e-4)

            # Update parameters as in WebDisco using lifelines stepsizer
            true_inv_h_dot_g_T = spsolve(
                -webdisco_hessian,
                webdisco_gradient.reshape((-1, 1)),
                assume_a="pos",
                check_finite=False,
            )
            norm_delta = norm(true_inv_h_dot_g_T)
            if self.learning_rate_strategy == "lifelines":
                step_size = stepsizer.update(norm_delta).next()
            else:
                step_size = 1.0

            updates = step_size * true_inv_h_dot_g_T
            weight_manager.increment_parameters(
                model=coxmodel,
                updates=[torch.from_numpy(updates.reshape((1, -1)))],
                with_batch_norm_parameters=False,
            )

    @pytest.mark.slow
    def test_descent(self):
        """Test descent."""
        # We measure the accuracy of the final fit
        (
            self.hessians,
            self.lls,
            self.final_params_list,
            self.computed_stds_list,
            _,
        ) = get_final_cox_model_function(
            self.ds_client,
            self.webdisco_experiment.compute_plan_keys[0].key,
            self.NUM_ROUNDS,
            self.standardize_data,
            self._duration_col,
            self._event_col,
        )
        m = copy.deepcopy(self.model)
        # We unnormalize the weights as self.X is normalized
        m.fc1.weight.data = torch.from_numpy(
            self.final_params_list[0] * self.computed_stds_list[0].to_numpy()
        )
        m.eval()
        with torch.no_grad():
            ypred = m(torch.from_numpy(self.X)).detach().numpy()
        # Penalizer and propensity score affect ability to achieve good C-index
        # or to retrieve exact original Cox coeff
        if (not self.use_propensity) and np.allclose(self.penalizer, 0.0):
            # We validate the fit wrt the c-index of hazard ratios ranking
            assert (
                concordance_index(np.abs(self.t), -ypred, self.E) > 0.85
            ), "WebDiscoTorch model could not rank the pairs well enough."
            # We validate the fit wrt the real Cox model used to generate the data
            assert (
                np.linalg.norm(m.fc1.weight.data.numpy().squeeze() - self.coeffs) < 1.0
            ), "WebDiscoTorch could not retrieve the true Cox model."

        # We match lifelines in all cases (including use of propensity)
        # except when there is a penalizer as lifelines
        # does like 50 iterations, which would take days in Substra
        # if we could afford to change cls.NUM_ROUNDS to say 60 isntead of 8
        # we could get rid of the if
        if np.allclose(self.penalizer, 0.0):
            assert np.allclose(self.cphf.params_, self.final_params_list[0], atol=1e-4)

    @pytest.mark.slow
    def test_standard_deviations(self):
        """Test standard deviations."""
        (
            self.hessians,
            self.lls,
            self.final_params_list,
            self.computed_stds_list,
            _,
        ) = get_final_cox_model_function(
            self.ds_client,
            self.webdisco_experiment.compute_plan_keys[0].key,
            self.NUM_ROUNDS,
            self.standardize_data,
            self._duration_col,
            self._event_col,
        )
        self.scaled_variance_matrix = -inv(self.hessians[0]) / np.outer(
            self.computed_stds_list[0], self.computed_stds_list[0]
        )
        summary = compute_summary_function(
            self.final_params_list[0], self.scaled_variance_matrix, self.cphf.alpha
        )
        ground_truth_df = self.cphf.summary
        ground_truth_df = ground_truth_df[summary.columns]
        # In case index are not matching
        summary.index = ground_truth_df.index
        gt_ll = self.cphf.log_likelihood_
        assert np.allclose(self.lls[0].item(), gt_ll)
        pd.testing.assert_frame_equal(
            summary, ground_truth_df, check_names=False, atol=1e-03
        )


class TestWebDiscoUnstandardized(TestWebDisco):
    """Test for unstandardized web disco."""

    @classmethod
    def setUpClass(cls):
        """Set up class."""
        super().setUpClass(standardize_data=False)


class TestWebDiscoWithWeights(TestWebDisco):
    """Test for weighted web disco."""

    @classmethod
    def setUpClass(cls):
        """Set up class."""
        super().setUpClass(use_propensity=True)

    def test_compute_true_moments(self):
        """Test computation of moments."""
        pass

    def test_aggregate_statistics(self):
        """Test aggregated statistics."""
        pass


class TestWebDiscoWithPenalizer(TestWebDisco):
    """Tests web disco with penalizers."""

    @classmethod
    def setUpClass(cls):
        """Set up class."""
        super().setUpClass(penalizer=0.1, l1_ratio=0.5)

    def test_compute_true_moments(self):
        """Test computation of moments."""
        pass

    def test_aggregate_statistics(self):
        """Test aggregated statistics."""
        pass

    def test_standard_deviations(self):
        """Test standard deviations."""
        # It's a pity but lifelines just does too many iterations to be able to test it
        # in a reasonable amount of time due to the slowness of this implementation
        # TODO test it in simu mode
        pass
