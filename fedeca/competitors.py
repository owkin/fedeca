"""Implementation of competitors of FEDECA."""
from typing import Literal, Optional

import numpy as np
import pandas as pd
from indcomp import MAIC
from lifelines.fitters.coxph_fitter import CoxPHFitter
from sklearn.linear_model import LogisticRegression

from fedeca.utils.survival_utils import (
    BaseSurvivalEstimator,
    BootstrapMixin,
    compute_summary,
)
from fedeca.utils.typing import _SeedType


class PooledIPTW(BaseSurvivalEstimator, BootstrapMixin):
    """Class for the Pooled IPTW."""

    def __init__(
        self,
        treated_col="treated",
        event_col="E",
        duration_col="T",
        ps_col="propensity_scores",
        effect="ATE",
        variance_method: Literal["naive", "robust", "bootstrap"] = "naive",
        n_bootstrap: int = 200,
        seed: _SeedType = None,
        cox_fit_kwargs=None,
    ):
        """Initialize Pooled Inverse Probability of Treatment Weighting estimator.

        Parameters
        ----------
        treated_col : str, optional
            Column name indicating treatment status, by default "treated".
        event_col : str, optional
            Column name indicating event occurrence, by default "E".
        duration_col : str, optional
            Column name indicating time to event or censoring, by default "T".
        ps_col : str, optional
            Column name indicating the propensity scores.
        effect : str, optional
            Effect type to estimate (ATE, ATC, or ATT), by default "ATE".
        variance_method : `{"naive", "robust", "bootstrap"}`
            Method for estimating the variance, and therefore the p-value of the
            estimated treatment effect.
            * "naive": Inverse of the Fisher information.
            * "robust": The robust sandwich estimator. Useful when samples are
              reweighted.
            * "bootstrap": Bootstrap the given data by sampling each patient
              with replacement, each time estimate the treatment effect, then
              use all repeated estimations to compute the variance.
        n_bootstrap : int
            Number of bootstrap repetitions, only useful when `variance_method`
            is set to "bootstrap", by default 200, as recommended in "Efron B,
            Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall:
            New York, NY, 1993, (page 52)."
        seed: {None, int, Sequence[int], SeedSequence, BitGenerator, Generator}
            The seed for reproducibility, only useful when `variance_method` is
            set to "bootstrap", by default None.
        cox_fit_kwargs : dict or None, optional
            Additional keyword arguments for Cox model fitting, by default None.
        """
        super().__init__(treated_col, event_col, duration_col, ps_col, seed)
        self.effect = effect
        if cox_fit_kwargs is None:
            cox_fit_kwargs = {}
        self.cox_fit_kwargs = cox_fit_kwargs
        self.variance_method = variance_method
        # cox_fit_kwargs takes priority
        if variance_method == "naive":
            self.cox_fit_kwargs.setdefault("robust", False)
        elif variance_method == "robust":
            self.cox_fit_kwargs.setdefault("robust", True)
        self.n_bootstrap = n_bootstrap

    def _estimate_effect(self, data: pd.DataFrame, weights: np.ndarray):
        """Estimate treatment effect."""
        # Estimate the effect from a weighted cox model
        # -> Estimand is the hazard ratio
        cox_model = CoxPHFitter()
        cox_model.fit(
            data[[self.duration_col, self.event_col, self.treated_col]].assign(
                weights=weights
            ),
            self.duration_col,
            self.event_col,
            weights_col="weights",
            **self.cox_fit_kwargs,
        )
        return cox_model

    def _fit(
        self, data: pd.DataFrame, targets: Optional[pd.DataFrame] = None
    ) -> tuple[pd.DataFrame, float, np.ndarray, np.ndarray]:
        """Estimate the treatment effect via Inverse Probability Treatment Weighting.

        targets: pd.DataFrame, optional
            pre-computed propensity scores.
            It is possible to pass pre-computed propensity scores to the fit
            function to use in the IPTW estimator instead of estimating the
            scores using logistic regression.
        """
        if targets is None:
            # Fit a logistic regression model to predict treatment assignment
            #  based on the confounding variables
            non_cov = [
                self.treated_col,
                self.event_col,
                self.duration_col,
                self.ps_col,
            ]
            covariates = [x for x in data.columns if x not in non_cov]
            X = data[covariates]
            logreg = LogisticRegression(solver="lbfgs", penalty=None)  # type: ignore
            logreg.fit(X, data[self.treated_col])

            # Compute the inverse probability weights
            prob = logreg.predict_proba(X)[:, 1]
        else:
            prob = targets.to_numpy().flatten()

        treated = data[self.treated_col] == 1
        control = data[self.treated_col] == 0
        weights = np.zeros_like(prob)
        if self.effect == "ATE":
            weights[treated] = np.divide(1, prob[treated])
            weights[control] = np.divide(1, 1 - prob[control])

        elif self.effect == "ATT":
            weights = data[self.treated_col]
            weights += prob * (1 - data[self.treated_col]) / (1 - prob)

        results = self._estimate_effect(data, weights)
        return results.summary, results.log_likelihood_, weights, prob

    def point_estimate(self, data: pd.DataFrame) -> np.ndarray:
        """Return a point estimate of the treatment effect."""
        results, _, _, _ = self._fit(data)
        return results["coef"]

    def fit(self, data: pd.DataFrame, targets: Optional[pd.DataFrame] = None):
        """Estimate the treatment effect via Inverse Probability Treatment Weighting.

        Option to estimate the variance of estimation by bootstrapping.

        targets: pd.DataFrame, optional
            pre-computed propensity scores.
            It is possible to pass pre-computed propensity scores to the fit
            function to use in the IPTW estimator instead of estimating the
            scores using logistic regression.
        """
        self.reset_state()

        results, loglik, weights, ps_scores = self._fit(data, targets)

        if self.variance_method == "bootstrap":
            std = self.bootstrap_std(data, self.n_bootstrap, self.rng)
            if std is not None:
                results = compute_summary(results["coef"], std, index=results.index)

        self.results_ = results
        self.log_likelihood_ = loglik
        self.propensity_scores_ = ps_scores
        self.weights_ = weights


# This is a wrapper for an existing implementation
# that can be found here https://github.com/AidanCooper/indcomp
# We have added the possibility to have more than one centers for
# which aggregated data is available and implement the method from
# Bucher et al., The results of direct and indirect treatment comparisons in
# meta-analysis of randomized controlled trials, (1997)


class MatchingAjudsted(BaseSurvivalEstimator, BootstrapMixin):
    """Implement Matching-Adjusted Indirect Comparisons class.

    We consider that we have access to individual patients data for one of the centers
    and that for the other centers we only have access to aggregated data. This method
    proposes a way to balance the distribution of the indivual patients data to match
    the mean (and std) of a list of covariates available in both studies.
    """

    def __init__(
        self,
        treated_col="treated",
        event_col="E",
        duration_col="T",
        ps_col="propensity_scores",
        variance_method: Literal["naive", "robust", "bootstrap"] = "naive",
        n_bootstrap: int = 200,
        seed: _SeedType = None,
        cox_fit_kwargs=None,
    ):
        """Initialize Matching-Adjusted Indirect Comparisons estimator.

        Parameters
        ----------
        treated_col : str, optional
            Column name indicating treatment status, by default "treated".
        event_col : str, optional
            Column name indicating event occurrence, by default "E".
        duration_col : str, optional
            Column name indicating time to event or censoring, by default "T".
        ps_col : str, optional
            Column name indicating propensity scores, by default "propensity_scores".
        variance_method : `{"naive", "robust", "bootstrap"}`
            Method for estimating the variance, and therefore the p-value of the
            estimated treatment effect.
            * "naive": Inverse of the Fisher information.
            * "robust": The robust sandwich estimator. Useful when samples are
              reweighted.
            * "bootstrap": Bootstrap the given data, each time estimate the
              treatment effect, then use all repeated estimations to compute the
              variance.
        n_bootstrap : int
            Number of bootstrap repetitions, only useful when `variance_method`
            is set to "bootstrap", by default 200, as recommended in "Efron B,
            Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall:
            New York, NY, 1993, (page 52)."
        seed: {None, int, Sequence[int], SeedSequence, BitGenerator, Generator}
            The seed for reproducibility, only useful when `variance_method` is
            set to "bootstrap", by default None.
        cox_fit_kwargs : dict or None, optional
            Additional keyword arguments for Cox model fitting, by default None.
        """
        super().__init__(treated_col, event_col, duration_col, ps_col, seed)
        if cox_fit_kwargs is None:
            cox_fit_kwargs = {}
        self.cox_fit_kwargs = cox_fit_kwargs
        self.variance_method = variance_method
        # cox_fit_kwargs takes priority
        if variance_method == "naive":
            self.cox_fit_kwargs.setdefault("robust", False)
        elif variance_method == "robust":
            self.cox_fit_kwargs.setdefault("robust", True)
        self.n_bootstrap = n_bootstrap

    # Implementation for only 2 trials for now
    # one with IPD and one with aggregated data

    def _fit(
        self, data: pd.DataFrame, targets: Optional[pd.DataFrame] = None
    ) -> tuple[pd.DataFrame, float, np.ndarray]:
        """Fit with reweighting on selected independent patient data.

        Parameters
        ----------
        data: pd.DataFrame
            Time-to-event datasets with part of rows to be reweighted before
            the estimation of treatment effect.

        targets: pd.DataFrame, optional
            Dataframe containing marginal statistics of covariates in `data` to
            be matched by reweighting. If None, assume a scenario grouped by
            `treated_col` in `data`, rows with "treated_col == 0" will be used
            to compute the marginal statistics.
        """
        non_cov = [
            self.treated_col,
            self.event_col,
            self.duration_col,
            self.ps_col,
        ]
        covariates = [x for x in data.columns if x not in non_cov]
        matching_dict = {}
        for col in covariates:
            matching_dict[col + ".mean"] = ("mean", col)
            matching_dict[col + ".std"] = ("std", col, col + ".mean")
        if targets is None:
            df_agg = data.groupby(self.treated_col)[covariates].agg(["mean", "std"])
            df_agg.columns = [".".join(x) for x in df_agg.columns]
            targets = pd.DataFrame(df_agg.loc[[0]])

        m_reweight = data[self.treated_col].ne(0)
        maic_model = MAIC(
            df_index=data[m_reweight],
            df_target=targets,
            match=matching_dict,
        )
        maic_model.calc_weights()

        weights = np.repeat(1.0, data.shape[0])
        weights[data.index[m_reweight]] = maic_model.weights_
        weights[weights <= 0] = 0.01

        cox_model = CoxPHFitter()

        cox_model.fit(
            data[[self.duration_col, self.event_col, self.treated_col]].assign(
                weights=weights
            ),
            self.duration_col,
            self.event_col,
            weights_col="weights",
            **self.cox_fit_kwargs,
        )

        return cox_model.summary, cox_model.log_likelihood_, weights

    def bootstrap_sample(
        self, data: pd.DataFrame, seed: _SeedType = None
    ) -> pd.DataFrame:
        """Resampling only the individual patient data (IPD) with replacement.

        In the setting of an estimation using MAIC, the caller is suppposed to have
        access only to the individual patient data, assumed here to be marked by non-
        zero treatment allocations in the data. Therefore during the resampling, only
        accessible data should be resampled.
        """
        rng = np.random.default_rng(seed)
        is_ipd = data[self.treated_col].ne(0)
        # resample individual patient data and concatenate with the rest
        data_resampled = data.loc[
            np.concatenate(
                [
                    rng.choice(data.index[is_ipd], size=is_ipd.sum(), replace=True),
                    data.index[~is_ipd],
                ]
            )
        ]
        return data_resampled

    def point_estimate(self, data: pd.DataFrame) -> np.ndarray:
        """Return a point estimate of the treatment effect."""
        results, _, _ = self._fit(data)
        return results["coef"]

    def fit(self, data: pd.DataFrame, targets: Optional[pd.DataFrame] = None) -> None:
        """Estimate the treatment effect via Inverse Probability Treatment Weighting.

        Option to estimate the variance of estimation by bootstrapping.

        targets: pd.DataFrame, optional
            pre-computed propensity scores.
            It is possible to pass pre-computed propensity scores to the fit
            function to use in the IPTW estimator instead of estimating the
            scores using logistic regression.
        """
        self.reset_state()

        results, loglik, weights = self._fit(data, targets)

        if self.variance_method == "bootstrap":
            std = self.bootstrap_std(data, self.n_bootstrap, self.rng)
            if std is not None:
                results = compute_summary(results["coef"], std, index=results.index)

        self.results_ = results
        self.log_likelihood_ = loglik
        self.weights_ = weights


class NaiveComparison(BaseSurvivalEstimator, BootstrapMixin):
    """Naive comparison as if in a randomized setting."""

    def __init__(
        self,
        treated_col="treated",
        event_col="E",
        duration_col="T",
        ps_col="propensity_scores",
        variance_method: Literal["naive", "robust", "bootstrap"] = "naive",
        n_bootstrap: int = 200,
        seed: _SeedType = None,
        cox_fit_kwargs=None,
    ):
        """Initialize Naive Comparison survival estimator.

        Parameters
        ----------
        treated_col : str, optional
            Column name indicating treatment status, by default "treated".
        event_col : str, optional
            Column name indicating event occurrence, by default "E".
        duration_col : str, optional
            Column name indicating time to event or censoring, by default "T".
        ps_col : str, optional
            Column name indicating the propensity scores.
        variance_method : `{"naive", "robust", "bootstrap"}`
            Method for estimating the variance, and therefore the p-value of the
            estimated treatment effect.
            * "naive": Inverse of the Fisher information.
            * "robust": The robust sandwich estimator. Useful when samples are
              reweighted.
            * "bootstrap": Bootstrap the given data, each time estimate the
              treatment effect, then use all repeated estimations to compute the
              variance.
        n_bootstrap : int
            Number of bootstrap repetitions, only useful when `variance_method`
            is set to "bootstrap", by default 200, as recommended in "Efron B,
            Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall:
            New York, NY, 1993, (page 52)."
        seed: {None, int, Sequence[int], SeedSequence, BitGenerator, Generator}
            The seed for reproducibility, only useful when `variance_method` is
            set to "bootstrap", by default None.
        cox_fit_kwargs : dict or None, optional
            Additional keyword arguments for Cox model fitting, by default None.
        """
        super().__init__(treated_col, event_col, duration_col, ps_col)
        if cox_fit_kwargs is None:
            cox_fit_kwargs = {}
        self.cox_fit_kwargs = cox_fit_kwargs
        self.variance_method = variance_method
        # cox_fit_kwargs takes priority
        if variance_method == "naive":
            self.cox_fit_kwargs.setdefault("robust", False)
        elif variance_method == "robust":
            self.cox_fit_kwargs.setdefault("robust", True)
        self.n_bootstrap = n_bootstrap

    def _fit(
        self, data: pd.DataFrame, targets: Optional[pd.DataFrame] = None
    ) -> tuple[pd.DataFrame, float, np.ndarray]:
        """Fit Naive Comparison estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Input data as a DataFrame.
        targets : pd.DataFrame, optional
            Target values associated with the input data, by default None.
            In the current implementation targets argument is not used
            by the fit function but is needed for the parent class.
        """
        cox_model = CoxPHFitter()
        cox_model.fit(
            data[[self.duration_col, self.event_col, self.treated_col]],
            self.duration_col,
            self.event_col,
            **self.cox_fit_kwargs,
        )
        weights = np.repeat(1, data.shape[0])
        return cox_model.summary, cox_model.log_likelihood_, weights

    def point_estimate(self, data: pd.DataFrame) -> np.ndarray:
        """Return a point estimate of the treatment effect."""
        results, _, _ = self._fit(data)
        return results["coef"]

    def fit(self, data: pd.DataFrame, targets: Optional[pd.DataFrame] = None) -> None:
        """Estimate the treatment effect via Inverse Probability Treatment Weighting.

        Option to estimate the variance of estimation by bootstrapping.

        targets: pd.DataFrame, optional
            pre-computed propensity scores.
            It is possible to pass pre-computed propensity scores to the fit
            function to use in the IPTW estimator instead of estimating the
            scores using logistic regression.
        """
        self.reset_state()

        results, loglik, weights = self._fit(data, targets)

        if self.variance_method == "bootstrap":
            std = self.bootstrap_std(data, self.n_bootstrap, self.rng)
            if std is not None:
                results = compute_summary(results["coef"], std, index=results.index)

        self.results_ = results
        self.log_likelihood_ = loglik
        self.weights_ = weights


class CovariateAdjusted(BaseSurvivalEstimator, BootstrapMixin):
    """Covariates adjusted IPTW."""

    def __init__(
        self,
        treated_col="treated",
        event_col="E",
        duration_col="T",
        ps_col="propensity_scores",
        variance_method: Literal["naive", "robust", "bootstrap"] = "naive",
        n_bootstrap: int = 200,
        seed: _SeedType = None,
        cox_fit_kwargs=None,
    ):
        """Initialize Covariate-Adjusted survival estimator.

        Parameters
        ----------
        treated_col : str, optional
            Column name indicating treatment status, by default "treated".
        event_col : str, optional
            Column name indicating event occurrence, by default "E".
        duration_col : str, optional
            Column name indicating time to event or censoring, by default "T".
        ps_col : str, optional
            Column name indicating propensity scores, by default "propensity_scores".
        variance_method : `{"naive", "robust", "bootstrap"}`
            Method for estimating the variance, and therefore the p-value of the
            estimated treatment effect.
            * "naive": Inverse of the Fisher information.
            * "robust": The robust sandwich estimator. Useful when samples are
              reweighted.
            * "bootstrap": Bootstrap the given data, each time estimate the
              treatment effect, then use all repeated estimations to compute the
              variance.
        n_bootstrap : int
            Number of bootstrap repetitions, only useful when `variance_method`
            is set to "bootstrap", by default 200, as recommended in "Efron B,
            Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall:
            New York, NY, 1993, (page 52)."
        seed: {None, int, Sequence[int], SeedSequence, BitGenerator, Generator}
            The seed for reproducibility, only useful when `variance_method` is
            set to "bootstrap", by default None.
        cox_fit_kwargs : dict or None, optional
            Additional keyword arguments for Cox model fitting, by default None.
        """
        super().__init__(treated_col, event_col, duration_col, ps_col)
        if cox_fit_kwargs is None:
            cox_fit_kwargs = {}
        self.cox_fit_kwargs = cox_fit_kwargs
        self.variance_method = variance_method
        # cox_fit_kwargs takes priority
        if variance_method == "naive":
            self.cox_fit_kwargs.setdefault("robust", False)
        elif variance_method == "robust":
            self.cox_fit_kwargs.setdefault("robust", True)
        self.n_bootstrap = n_bootstrap

    def _fit(
        self, data: pd.DataFrame, targets: Optional[pd.DataFrame] = None
    ) -> tuple[pd.DataFrame, float, np.ndarray]:
        """Fit Covariate-Adjusted estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Input data as a DataFrame.
        targets : pd.DataFrame, optional
            Target values associated with the input data, by default None.
            In the current implementation targets argument is not used
            by the fit function but is needed for the parent class.
        """
        non_cov = [self.treated_col, self.event_col, self.duration_col, self.ps_col]
        covariates = [x for x in data.columns if x not in non_cov]
        cox_model = CoxPHFitter()
        cox_model.fit(
            data[[self.treated_col, self.event_col, self.duration_col] + covariates],
            self.duration_col,
            self.event_col,
            **self.cox_fit_kwargs,
        )
        weights = np.repeat(1, data.shape[0])
        return cox_model.summary, cox_model.log_likelihood_, weights

    def point_estimate(self, data: pd.DataFrame) -> np.ndarray:
        """Return a point estimate of the treatment effect."""
        results, _, _ = self._fit(data)
        return results["coef"]

    def fit(self, data: pd.DataFrame, targets: Optional[pd.DataFrame] = None) -> None:
        """Estimate the treatment effect via Inverse Probability Treatment Weighting.

        Option to estimate the variance of estimation by bootstrapping.

        targets: pd.DataFrame, optional
            pre-computed propensity scores.
            It is possible to pass pre-computed propensity scores to the fit
            function to use in the IPTW estimator instead of estimating the
            scores using logistic regression.
        """
        self.reset_state()

        results, loglik, weights = self._fit(data, targets)

        if self.variance_method == "bootstrap":
            std = self.bootstrap_std(data, self.n_bootstrap, self.rng)
            if std is not None:
                results = compute_summary(results["coef"], std, index=results.index)

        self.results_ = results
        self.log_likelihood_ = loglik
        self.weights_ = weights
