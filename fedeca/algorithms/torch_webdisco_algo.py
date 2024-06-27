"""Implement webdisco algorithm with Torch."""
import copy
from copy import deepcopy
from math import sqrt
from pathlib import Path
from typing import Any, List, Optional, Union

# hello
import numpy as np
import torch
from autograd import elementwise_grad
from autograd import numpy as anp
from lifelines.utils import StepSizer
from pandas.api.types import is_numeric_dtype
from scipy.linalg import norm
from scipy.linalg import solve as spsolve
from substrafl.algorithms.pytorch import weight_manager
from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.remote import remote_data
from substrafl.strategies.schemas import StrategyName

from fedeca.schemas import WebDiscoAveragedStates, WebDiscoSharedState
from fedeca.utils.moments_utils import compute_uncentered_moment
from fedeca.utils.survival_utils import MockStepSizer


class TorchWebDiscoAlgo(TorchAlgo):
    """WebDiscoAlgo class."""

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: Optional[int],
        *args,
        duration_col: str = "T",
        event_col: str = "E",
        treated_col: str = None,
        initial_step_size: float = 0.95,
        learning_rate_strategy: str = "lifelines",
        standardize_data: bool = True,
        tol: float = 1e-16,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
        propensity_model: torch.nn.Module = None,
        training_strategy: str = "iptw",
        cox_fit_cols: Union[None, list] = None,
        propensity_fit_cols: Union[None, list] = None,
        store_hessian: bool = False,
        with_batch_norm_parameters: bool = False,
        use_gpu: bool = True,
        robust: bool = False,
        **kwargs,
    ):
        """Initialize the TorchWebdiscoAlgo class.

        Parameters
        ----------
        model : torch.nn.Module
            Model to use internally
        batch_size : int, optional
            Batch size for training
        duration_col : str, optional
            Column for the duration. Defaults to "T".
        event_col : str, optional
            Column for the event. Defaults to "E".
        treated_col : str, optional
            Column for the treatment. Defaults to None.
        initial_step_size : float, optional
            Initial step size. Defaults to 0.95.
        learning_rate_strategy : str, optional
            Strategy to follow for the learning rate. Defaults to "lifelines".
        standardize_data : bool, optional
            Whether to standardize data. Defaults to True.
        tol : float, optional
            Precision tolerance. Defaults to 1e-16.
        penalizer : float, optional
            Strength of the total penalization. Defaults to 0.0.
        l1_ratio : float, optional
            Ratio of the L1 penalization, should be in [0, 1]. Defaults to 0.0.
        propensity_model : torch.nn.Module, optional
            Propensity model to use. Defaults to None.
        training_strategy : str, optional
            Which covariates to use for the Cox model.
            Both give different results because of non-collapsibility:
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7986756/
            Defaults to iptw, which will use only the treatment allocation as covariate.
            Can be iptw, aiptw or webdisco. Aiptw will use both cox_fit_cols
            and treatment allocation and cox will use cox_fit_cols only.
        cox_fit_cols : Union[None, list], optional
            Columns to use for the Cox model's covariates. Defaults to None for
            IPTW analysis hence using only the treatment column or all columns
            if propensity strategy is Cox.
        propensity_fit_cols : Union[None, list], optional
            Columns to use for the propensity model's input. Defaults to None
            which means everything.
        store_hessian : bool, optional
            Whether to store the Hessian. Defaults to False.
        with_batch_norm_parameters : bool, optional
            Whether to use batch norm parameters. Defaults to False.
        use_gpu : bool, optional
            Whether to use GPU for training. Defaults to True.
        robust : bool, optional
            Whether or not to store quantities specifically needed for robust
            estimation.
        """
        assert "optimizer" not in kwargs, "WebDisco strategy does not uses optimizers"
        assert "criterion" not in kwargs, "WebDisco strategy does not use criterion"
        assert training_strategy in [
            "iptw",
            "aiptw",
            "webdisco",
        ], f"""propensity strategy not {training_strategy}
        Implemented"""

        super().__init__(
            model=model,
            criterion=None,
            optimizer=None,
            index_generator=None,
            use_gpu=use_gpu,
            # duration_col=duration_col,
            # batch_size=batch_size,
            # tol=tol,
            # initial_step_size=initial_step_size,
            # learning_rate_strategy=learning_rate_strategy,
            # penalizer=penalizer,
            # l1_ratio=l1_ratio,
            # with_batch_norm_parameters=with_batch_norm_parameters,
            **kwargs,
        )
        self._batch_size = batch_size
        self._duration_col = duration_col
        self._event_col = event_col
        self._target_cols = [self._duration_col, self._event_col]
        self._treated_col = treated_col if treated_col is not None else []
        self._standardize_data = standardize_data
        self._tol = tol
        self._initial_step_size = initial_step_size
        assert learning_rate_strategy in [
            "lifelines",
            "constant",
        ], "Learning rate strategy not supported"
        self._learning_rate_strategy = learning_rate_strategy
        self._penalizer = penalizer
        self._l1_ratio = l1_ratio
        self._propensity_model = propensity_model
        if self._propensity_model is not None:
            assert (
                self._treated_col is not None
            ), "If you are using a propensity model you should provide the treated col"
            self._propensity_model.eval()
        self._training_strategy = training_strategy

        self._propensity_fit_cols = propensity_fit_cols
        self._cox_fit_cols = cox_fit_cols if cox_fit_cols is not None else []

        self._store_hessian = store_hessian
        self._with_batch_norm_parameters = with_batch_norm_parameters
        self._robust = robust

        self.server_state = {}
        self.global_moments = None
        # initialized and used only in the train method
        self._final_gradients = None
        self._final_hessian = None
        self._n_samples_done = None

        # TODO make this as clean as possible but frankly it's hard
        # you want wargs to be simultaneously empty and not empty
        for k in ["propensity_model", "robust"]:
            self.kwargs[k] = copy.deepcopy(getattr(self, "_" + k))

    @property
    def strategies(self) -> List[StrategyName]:
        """List of compatible strategies.

        Returns
        -------
        typing.List[StrategyName]
            List of compatible strategies.
        """
        return ["WebDisco"]

    @remote_data
    def compute_local_phi_stats(
        self,
        data_from_opener: Any,
        # Set shared_state to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
        shared_state: Optional[WebDiscoAveragedStates] = None,
    ) -> WebDiscoSharedState:
        """Compute local updates.

        Parameters
        ----------
        data_from_opener : Any
            _description_
        shared_state : Optional[WebDiscoAveragedStates], optional
            _description_. Defaults to None.

        Returns
        -------
        WebDiscoSharedState
            _description_
        """
        if not hasattr(self, "server_state"):
            self.server_state = {}
        # We either already have global_statistics in the self or we don't
        if shared_state is None:
            # This is part of the stateless server
            global_survival_statistics = self.server_state["global_survival_statistics"]
            # We assume moments have already been given once to each client
            # so that they updated their self.global_moments if standardize_data
            # is true so moments can be None
            moments = None
        else:
            # We initialize the self.server_state["global_survival_statistics"]
            # that will be used
            # throughout
            global_survival_statistics = shared_state["global_survival_statistics"]
            self.server_state["global_survival_statistics"] = global_survival_statistics
            moments = shared_state["moments"]

        X, y, weights = self.compute_X_y_and_propensity_weights(
            data_from_opener, moments
        )

        distinct_event_times = global_survival_statistics["distinct_event_times"]

        self._model.eval()
        # The shape of expbetaTx is (N, 1)
        X = torch.from_numpy(X)
        expbetaTx = self._model(X).detach().numpy()
        X = X.numpy()
        betaTx = np.log(expbetaTx)  # very inefficient, but whatever
        offset = betaTx.max(axis=0)
        factor = np.exp(offset)
        expbetaTx_stable = np.exp(betaTx - offset)
        # for risk_phi each element is a scalar
        risk_phi = []
        # for risk_phi_x each element is of the dimension of a feature N,
        risk_phi_x = []
        # for risk_phi_x_x each element is of the dimension of a feature squared N, N
        risk_phi_x_x = []
        for _, t in enumerate(distinct_event_times):
            Rt = np.where(np.abs(y) >= t)[0]
            weights_for_rt = weights[Rt]
            risk_phi.append(
                factor
                * (np.multiply(expbetaTx_stable[Rt], weights_for_rt).sum(axis=(0, 1)))
            )
            common_block = np.multiply(expbetaTx_stable[Rt] * weights_for_rt, X[Rt])
            risk_phi_x.append(factor * common_block.sum(axis=0))
            risk_phi_x_x.append(factor * np.einsum("ij,ik->jk", common_block, X[Rt]))
        local_phi_stats = {}
        local_phi_stats["risk_phi"] = risk_phi
        local_phi_stats["risk_phi_x"] = risk_phi_x
        local_phi_stats["risk_phi_x_x"] = risk_phi_x_x

        return {
            "local_phi_stats": local_phi_stats,
            # The server being stateless we need to feed it perpetually
            "global_survival_statistics": global_survival_statistics,
        }

    @remote_data
    def local_uncentered_moments(self, data_from_opener, shared_state=None):
        """Compute the local uncentered moments.

        This method is transformed by the decorator to meet Substra API,
        and is executed in the training nodes. See build_compute_plan.

        Parameters
        ----------
        data_from_opener : pd.DataFrame
            Dataframe returned by the opener.
        shared_state : None, optional
            Given by the aggregation node, here nothing, by default None.

        Returns
        -------
        dict
            Local results to be shared via shared_state to the aggregation node.
        """
        del shared_state  # unused
        # We do not have to do the mean on the target columns
        data_from_opener = data_from_opener.drop(columns=self._target_cols)
        if self._propensity_model is not None:
            assert self._treated_col is not None
            if self._training_strategy == "iptw":
                data_from_opener = data_from_opener.loc[:, [self._treated_col]]
            elif self._training_strategy == "aiptw":
                data_from_opener = data_from_opener.loc[
                    :, [self._treated_col] + self._cox_fit_cols
                ]
        else:
            assert self._training_strategy == "webdisco"
            if len(self._cox_fit_cols) > 0:
                data_from_opener = data_from_opener.loc[:, self._cox_fit_cols]
            else:
                pass

        results = {
            f"moment{k}": compute_uncentered_moment(data_from_opener, k)
            for k in range(1, 3)
        }
        results["n_samples"] = data_from_opener.select_dtypes(include=np.number).count()
        return results

    @remote_data
    def _compute_local_constant_survival_statistics(
        self, data_from_opener, shared_state
    ):
        """Computes local statistics and Dt for all ts in the distinct event times.
        Those statistics are useful for to compute the global statistics that will be
        used throughout training. The definition of :math:`\\mathcal{D}_t` (Dt)
        associated to the value t is the set of indices of all the individuals that
        experience an event at time t.

        More formally:

        .. math::

            \\mathcal{D}_{t} = \{ i \in [0, n] | e_i = 0, t_i = t\}  # noqa W630



        Parameters
        ----------
        tokens_list : list
            Normally a list of size one since we should use all samples in one batch.

        Returns
        -------
        dict
            Where we can find the following keys 'sum_features_on_events',
            'distinct_event_times', 'number_events_by_time' and 'total_number_samples',
            where:
            - "sum_features_on_events" contains the sum of the features
                across samples for all the distinct event times of the given clients,
                i.e. a single vector per time stamp
            - "distinct_event_times": list of floating values containing the
                unique times at which at least 1 death is registered in the
                current dataset
            - "number_events_by_time": number of events occurring at each
            distinct_event_times
            - "total_number_samples": total number of samples
        """
        X, y, weights = self.compute_X_y_and_propensity_weights(
            data_from_opener, shared_state
        )
        distinct_event_times = np.unique(y[y > 0]).tolist()

        sum_features_on_events = np.zeros(X.shape[1:])
        number_events_by_time = []
        weights_counts_on_events = []
        for t in distinct_event_times:
            Dt = np.where(y == t)[0]
            num_events = len(Dt)
            sum_features_on_events += (weights[Dt] * X[Dt, :]).sum(axis=0)
            number_events_by_time.append(num_events)
            weights_counts_on_events.append(weights[Dt].sum())

        return {
            "sum_features_on_events": sum_features_on_events,
            "distinct_event_times": distinct_event_times,
            "number_events_by_time": number_events_by_time,
            "total_number_samples": X.shape[0],
            "moments": shared_state,
            "weights_counts_on_events": weights_counts_on_events,
        }

    @remote_data
    def train(
        self,
        data_from_opener: Any,
        # Set shared_state to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
        shared_state: Optional[WebDiscoAveragedStates] = None,
    ) -> WebDiscoSharedState:
        """Local train function.

        Parameters
        ----------
        data_from_opener : Any
            _description_
        shared_state  : Optional[WebDiscoAveragedStates], optional
            description_. Defaults to None.

        Raises
        ------
        NotImplementedError
            _description_

        Returns
        -------
        WebDiscoSharedState
            _description_
        """
        # We either simply update the model with NR update or we compute risk_phi_stats
        gradient = shared_state["gradient"]
        hessian = shared_state["hessian"]
        second_part_ll = shared_state["second_part_ll"]
        global_survival_statistics = self.server_state["global_survival_statistics"]
        first_part_ll = deepcopy(
            global_survival_statistics["global_sum_features_on_events"]
        )

        if "step_sizer" not in self.server_state:
            if self._learning_rate_strategy == "lifelines":
                self.server_state["step_sizer"] = StepSizer(self._initial_step_size)
            else:
                # use constant learning rate of 1.
                self.server_state["step_sizer"] = MockStepSizer()
            self.server_state["count_iter"] = 1
            self.server_state["current_weights"] = np.zeros(
                shared_state["gradient_shape"]
            )

        n = global_survival_statistics["total_number_samples"]

        if self._penalizer > 0.0:
            if self._learning_rate_strategy == "lifelines":
                # This is used to multiply the penalty
                # We use a smooth approximation for the L1 norm (for more details
                # see docstring of function)
                # we use numpy autograd to be able to compute the first and second
                # order derivatives of this expression

                def soft_abs(x, a):
                    return 1 / a * (anp.logaddexp(0, -a * x) + anp.logaddexp(0, a * x))

                def elastic_net_penalty(beta, a):
                    l1 = self._l1_ratio * soft_abs(beta, a)
                    l2 = 0.5 * (1 - self._l1_ratio) * (beta**2)
                    reg = n * (self._penalizer * (l1 + l2)).sum()
                    return reg

                # Regularization affects both the gradient and the hessian
                # producing a better conditioned hessian.
                d_elastic_net_penalty = elementwise_grad(elastic_net_penalty)
                dd_elastic_net_penalty = elementwise_grad(d_elastic_net_penalty)
                # lifelines trick to progressively sharpen the approximation of
                # the l1 regularization.
                alpha = 1.3 ** self.server_state["count_iter"]
                # We are trying to **maximize** the log-likelihood that is why
                # we put a negative sign and not a plus sign on the regularization.
                # The fact that we are actually moving towards the maximum and
                # not towards the minimum is because -H is psd.
                gradient -= d_elastic_net_penalty(
                    self.server_state["current_weights"], alpha
                )
                hessian[
                    np.diag_indices(shared_state["gradient_shape"])
                ] -= dd_elastic_net_penalty(self.server_state["current_weights"], alpha)
            else:
                raise NotImplementedError

        inv_h_dot_g_T = spsolve(-hessian, gradient, assume_a="pos", check_finite=False)

        norm_delta = norm(inv_h_dot_g_T)

        step_size = self.server_state["step_sizer"].update(norm_delta).next()
        self.server_state["count_iter"] += 1
        updates = step_size * inv_h_dot_g_T

        # We keep the current version of the weights, because of ll computations
        past_ll = (self.server_state["current_weights"] * first_part_ll).sum(
            axis=0
        ) + second_part_ll
        self.server_state["current_weights"] += updates

        weight_manager.increment_parameters(
            model=self._model,
            updates=[torch.from_numpy(updates[None, :])],
            with_batch_norm_parameters=self._with_batch_norm_parameters,
        )

        # convergence criteria
        if norm_delta < 1e-07:
            converging, success = False, True
        elif step_size <= 0.00001:
            converging, success = False, False
        else:
            converging, success = True, False

        self.server_state["converging"] = converging
        self.server_state["success"] = success
        self.server_state["past_ll"] = past_ll
        # We store the hessian to compute standard deviations of coefficients and
        # associated p-values
        if self.server_state["count_iter"] > 10 or success or self._store_hessian:
            self.server_state["hessian"] = hessian
            self.server_state["gradient"] = gradient
        # This needs to be in the state of the client for complicated reasons due
        # to simu mode
        if self._robust:
            self.server_state["global_robust_statistics"] = {}
            self.server_state["global_robust_statistics"][
                "global_risk_phi_list"
            ] = shared_state["global_risk_phi_list"]
            self.server_state["global_robust_statistics"][
                "global_risk_phi_x_list"
            ] = shared_state["global_risk_phi_x_list"]
            # TODO this renaming and moving around is useless and inefficient
            self.server_state["global_robust_statistics"][
                "global_weights_counts_on_events"
            ] = self.server_state["global_survival_statistics"][
                "weights_counts_on_events"
            ]
            self.server_state["global_robust_statistics"][
                "distinct_event_times"
            ] = self.server_state["global_survival_statistics"]["distinct_event_times"]
        return self.compute_local_phi_stats(
            data_from_opener=data_from_opener, shared_state=None, _skip=True
        )

    def predict(
        self,
        data_from_opener: Any,
        shared_state: Any = None,
    ) -> Any:
        """Predict function.

        Execute the following operations:

            - Create the test torch dataset.
            - Execute and return the results of the ``self._local_predict`` method

        Parameters
        ----------
        data_from_opener : typing.Any
            Input data
        shared_state : typing.Any
            Latest train task shared state (output of the train method)
        """
        X, _, _ = self.compute_X_y_and_propensity_weights(
            data_from_opener, shared_state
        )

        X = torch.from_numpy(X)

        self._model.eval()

        predictions = self._model(X).cpu().detach().numpy()
        return predictions

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
        checkpoint.update({"server_state": self.server_state})
        checkpoint.update({"global_moments": self.global_moments})
        return checkpoint

    def _update_from_checkpoint(self, path: Path) -> dict:
        """Load the local state from the checkpoint.

        Parameters
        ----------
        checkpoint : dict
            The checkpoint to load.
        """
        checkpoint = super()._update_from_checkpoint(path=path)
        self.server_state = checkpoint.pop("server_state")
        self.global_moments = checkpoint.pop("global_moments")
        return checkpoint

    def summary(self):
        """Summary of the class to be exposed in the experiment summary file.

        Returns
        -------
        dict
            A json-serializable dict with the attributes the user wants to store
        """
        summary = super().summary()
        return summary

    def build_X_y(self, data_from_opener, shared_state={}):
        """Build appropriate X and y times from output of opener.

        This function 1. uses the event column to inject the censorship
        information present in the duration column (given in absolute values)
        in the form of a negative sign.
        2. Drop every covariate except treatment if self.strategy == "iptw".
        3. Standardize the data if self.standardize_data AND if it receives
        an outmodel.
        4. Return the (unstandardized) input to the propensity model Xprop if
        necessary as well as the treated column to be able to compute the
        propensity weights.

        Parameters
        ----------
        data_from_opener : pd.DataFrame
            The output of the opener
        shared_state : dict, optional
            Outmodel containing global means and stds.
            by default {}

        Returns
        -------
        tuple
            standardized X, signed times, treatment column and unstandardized
            propensity model input
        """
        # We need y to be in the format (2*event-1)*duration
        data_from_opener["time_multiplier"] = [
            2.0 * e - 1.0 for e in data_from_opener[self._event_col].tolist()
        ]
        # No funny business irrespective of the convention used
        y = (
            np.abs(data_from_opener[self._duration_col])
            * data_from_opener["time_multiplier"]
        )
        y = y.to_numpy().astype("float64")
        data_from_opener = data_from_opener.drop(columns=["time_multiplier"])
        # dangerous but we need to do it
        string_columns = [
            col
            for col in data_from_opener.columns
            if not (is_numeric_dtype(data_from_opener[col]))
        ]
        data_from_opener = data_from_opener.drop(columns=string_columns)

        # We drop the targets from X
        columns_to_drop = self._target_cols
        X = data_from_opener.drop(columns=columns_to_drop)
        if self._propensity_model is not None:
            assert self._treated_col is not None
            if self._training_strategy == "iptw":
                X = X.loc[:, [self._treated_col]]
            elif self._training_strategy == "aiptw":
                X = X.loc[:, [self._treated_col] + self._cox_fit_cols]
        else:
            assert self._training_strategy == "webdisco"
            if len(self._cox_fit_cols) > 0:
                X = X.loc[:, self._cox_fit_cols]
            else:
                pass

        # If X is to be standardized we do it
        if self._standardize_data:
            if shared_state:
                # Careful this shouldn't happen apart from the predict
                means = shared_state["global_uncentered_moment_1"]
                vars = shared_state["global_centered_moment_2"]
                # Careful we need to match pandas and use unbiased estimator
                bias_correction = (shared_state["total_n_samples"]) / float(
                    shared_state["total_n_samples"] - 1
                )
                self.global_moments = {
                    "means": means,
                    "vars": vars,
                    "bias_correction": bias_correction,
                }
                stds = vars.transform(lambda x: sqrt(x * bias_correction + self._tol))
                X = X.sub(means)
                X = X.div(stds)
            else:
                X = X.sub(self.global_moments["means"])
                stds = self.global_moments["vars"].transform(
                    lambda x: sqrt(
                        x * self.global_moments["bias_correction"] + self._tol
                    )
                )
                X = X.div(stds)

        X = X.to_numpy().astype("float64")

        # If we have a propensity model we need to build X without the targets AND the
        # treated column
        if self._propensity_model is not None:
            # We do not normalize the data for the propensity model !!!
            Xprop = data_from_opener.drop(columns=columns_to_drop + [self._treated_col])
            if self._propensity_fit_cols is not None:
                Xprop = Xprop[self._propensity_fit_cols]
            Xprop = Xprop.to_numpy().astype("float64")
        else:
            Xprop = None

        # If WebDisco is used without propensity treated column does not exist
        if self._treated_col is not None:
            treated = (
                data_from_opener[self._treated_col]
                .to_numpy()
                .astype("float64")
                .reshape((-1, 1))
            )
        else:
            treated = None

        return (X, y, treated, Xprop)

    def compute_X_y_and_propensity_weights(self, data_from_opener, shared_state):
        """Build appropriate X, y and weights from raw output of opener.

        Uses the helper function build_X_y and the propensity model to build the
        weights.

        Parameters
        ----------
        data_from_opener : pd.DataFrame
            Raw output from opener
        shared_state : dict, optional
            Outmodel containing global means and stds, by default {}

        Returns
        -------
        tuple
            _description_
        """
        X, y, treated, Xprop = self.build_X_y(data_from_opener, shared_state)
        if self._propensity_model is not None:
            assert (
                treated is not None
            ), f"""If you are using a propensity model the {self._treated_col} (Treated)
            column should be available"""
            assert np.all(
                np.in1d(np.unique(treated.astype("uint8"))[0], [0, 1])
            ), "The treated column should have all its values in set([0, 1])"
            Xprop = torch.from_numpy(Xprop)
            with torch.no_grad():
                propensity_scores = self._propensity_model(Xprop)

            propensity_scores = propensity_scores.detach().numpy()
            # We robustify the division
            weights = treated * 1.0 / np.maximum(propensity_scores, self._tol) + (
                1 - treated
            ) * 1.0 / (np.maximum(1.0 - propensity_scores, self._tol))
        else:
            weights = np.ones((X.shape[0], 1))
        return X, y, weights
