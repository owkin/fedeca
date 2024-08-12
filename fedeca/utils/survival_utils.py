"""Provide utils to simulate survival data."""
from __future__ import annotations

import copy
from math import sqrt
from typing import Final, Literal, Optional, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas.api.types import is_numeric_dtype
from scipy import stats
from scipy.linalg import toeplitz
from sklearn.base import BaseEstimator
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler

from fedeca.utils.typing import _FuncCateType, _FuncPropensityType, _SeedType


class BaseSurvivalEstimator(BaseEstimator):
    """Base estimator for time-to-event analysis."""

    def __init__(
        self,
        treated_col: str = "treated",
        event_col: str = "event",
        duration_col: str = "time",
        ps_col: Optional[str] = "propensity_scores",
        seed: _SeedType = None,
    ):
        """Initialize the BaseEstimator class.

        Parameters
        ----------
        treated_col : str, optional
            Column name indicating treatment status, by default "treated".
        event_col : str, optional
            Column name indicating event occurrence, by default "event".
        duration_col : str, optional
            Column name indicating time to event or censoring, by default "time".
        ps_col : str or None, optional
            Column name indicating propensity scores, by default "propensity_scores".
        seed: {None, int, Sequence[int], SeedSequence, BitGenerator, Generator}
            The seed for reproducibility. Defaults to None.
        """
        self.treated_col = treated_col
        self.event_col = event_col
        self.duration_col = duration_col
        self.ps_col = ps_col
        self.rng = np.random.default_rng(seed)
        self.log_likelihood_: Optional[float] = None
        self.results_: Optional[pd.DataFrame] = None
        self.weights_: Optional[np.ndarray] = None
        self.propensity_scores_: Optional[np.ndarray] = None

    def fit(self, data: pd.DataFrame, targets: Optional[pd.DataFrame] = None) -> None:
        """Fit the model to the provided data and optionally the target values.

        This method trains the model using the input data and, if available,
        the target values. The model's internal parameters are updated during
        training to learn from the provided data.
        The updated internal parameters are:
        * results_: summary data frame of the fitting results. Expect at least
          one row labeled by `self.treated_col`, with columns "coef" for
          treatment effect, and column "p" for the p-value of the estimation.
        * weights_: weights assigned to each row of `data`
        * log_likelihood_: The log-likelihood of the model fitted with `data`

        Parameters
        ----------
        data : pd.DataFrame
            Input data as a DataFrame containing features used for training the model.
        targets : pd.DataFrame, optional
            Target values associated with the input data, by default None.
            If provided, the model is trained using both the input data
            and target values.

        Returns
        -------
        None
        """

    def reset_state(self) -> None:
        """Reset the estimator's internal parameters related to fitted results."""
        self.results_ = None
        self.weights_ = None
        self.log_likelihood_ = None
        self.propensity_scores_ = None

    def set_random_state(self, seed: _SeedType) -> None:
        """Set random state."""
        self.rng = np.random.default_rng(seed)


class BootstrapMixinProtocol(Protocol):
    """Protocol class for type checking."""

    def point_estimate(self, data: pd.DataFrame) -> npt.ArrayLike:
        """Return a point estimate of treatment effect."""
        return np.array([])


class BootstrapMixin(BootstrapMixinProtocol):
    """Mixin class for bootstrapping utilities."""

    def bootstrap_sample(
        self, data: pd.DataFrame, seed: _SeedType = None
    ) -> pd.DataFrame:
        """Resampling with replacement."""
        rng = np.random.default_rng(seed)
        bootstrapped_data = data.sample(data.shape[0], replace=True, random_state=rng)
        return bootstrapped_data

    def bootstrap_std(
        self, data: pd.DataFrame, n_bootstrap: int, seed: _SeedType = None
    ) -> Optional[np.ndarray]:
        """Bootstrap the standard deviation of the treatment effect estimation."""
        if n_bootstrap <= 1:
            return None
        rng = np.random.default_rng(seed)

        def bootstrap_coef() -> np.ndarray:
            data_resampled = self.bootstrap_sample(data, rng)
            return np.array(self.point_estimate(data_resampled))

        final_params_array = [bootstrap_coef() for _ in range(n_bootstrap)]
        std = np.std(final_params_array, axis=0)
        return std


def compute_summary(
    coef: np.ndarray,
    coef_std: np.ndarray,
    coef_null: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    index: Optional[list[str] | pd.Index] = None,
) -> pd.DataFrame:
    """Compute summary for parameter estimation."""
    conf_int = f"{np.round(100 * (1 - alpha)):.0f}"
    delta_coef = np.multiply(stats.norm.ppf(1 - alpha / 2), coef_std)
    if coef_null is None:
        coef_null = np.zeros_like(coef)
    z_score = (coef - coef_null) / coef_std
    p_value = 2 * stats.norm.sf(np.abs(z_score))

    res = pd.DataFrame()
    res["coef"] = coef
    res["exp(coef)"] = np.exp(coef)
    res["se(coef)"] = coef_std
    res[f"coef lower {conf_int}%"] = coef - delta_coef
    res[f"coef upper {conf_int}%"] = coef + delta_coef
    res[f"exp(coef) lower {conf_int}%"] = np.exp(coef - delta_coef)
    res[f"exp(coef) upper {conf_int}%"] = np.exp(coef + delta_coef)
    res["cmp to"] = coef_null
    res["z"] = z_score
    res["p"] = p_value
    res["-log2(p)"] = -np.log2(p_value)

    if index is not None:
        res.index = index

    return res


class CoxData:
    """Simulate Cox data.

    This class simulates survival data following Cox model assumptions.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        ndim: int = 10,
        features_type: Literal[
            "cov_toeplitz",
            "cov_uniform",
            "indep_gauss",
        ] = "cov_toeplitz",
        cate: float | Literal["random", "linear"] = 1.0,
        propensity: Literal["constant", "linear"] = "constant",
        prop_treated: float = 0.5,
        overlap: float = 0.0,
        cov_corr: float = 0.5,
        scale_t: float = 1.0,
        shape_t: float = 1.0,
        censoring_factor: float = 0.5,
        percent_ties: Optional[float] = None,
        random_censoring: bool = False,
        seed: _SeedType = None,
        standardize_features: bool = True,
        dtype: Literal["float32", "float64"] = "float64",
    ):
        r"""Cox Data generator class.

        This class generates data according to a Cox proportional hazards model
        in continuous time as follows:
        .. math::
          S(t|x) = P(T > t | X=x)
          \\lambda(t|x) = \\frac{d \\log S(t|x)}{dt}
          \\lambda(t|x) = \\lambda_0(t)e^{\\beta^T x}
          \\Lambda_0(t|x) = \\int_0^t \\lambda_0(u)du = (\\frac{t}{s})^k
          X \\sim \\mathcal{N}(0, C)
          \\beta \\sim \\mathcal{N}(0, I)

        Parameters
        ----------
        n_samples: int, optional
            Number of samples to generate. Defaults to 1000
        ndim: int, optional
            Number of features, defaults to 10.
        features_type: `{"cov_toeplitz", "cov_uniform", "indep_gauss"}`, optional
        cate: {float, `{"random", "linear"}`, Callable}
            The way to assign treatment effect (hazard ratio) to samples.
            * "float": Constant hazard ratio for all samples.
            * "random": Hazard ratio follows log-normal distribution.
            * "linear": Hazard ratio depends on a linear combination of
              features with random coefficients.
            Defaults to 1.0 (no treatment effect).
        propensity: {`{"constant", "linear"}`, Callable}
            The way to assign propensity scores (probabilities of being treated)
            to samples.
            * "linear": Propensity scores depend on a linear combination of
              features with random coefficients.
            * "constant": All propensity scores take the value of the constant
              defined by the parameter `prop_treated`.
            Defaults to "constant".
        prop_treated: float, optional
            The proportion of the treated subpopulation. Defaults to 0.5.
        overlap: float, optional
            The strength of the covariate shift between control and treated.
            If 0. then the propensity is random. Defauls to 0.
        cov_corr: float, optional
            The correlation of the covariance matrix.
        scale_t: float, optional
            Scale parameter `s` in the equations above. Defaults to `1.0`.
        shape_t: float, optional
            Shape parameter `k` in the equations above. Defaults to `1.0`.
        censoring_factor: float, optional
            Parameter used to determine the probability of being censored
            (with respect to the median). Defaults to `0.5`.
        percent_ties: float, optional
            Parameter that control the percentage of samples who have the same outcome.
            Defaults to None.
        random_censoring: bool, optional
            Whether to censor completely independently of the rest or not.
            When true, censors samples with probability censoring_factor.
            When false, samples are censored if the drawn event times
            (drawn from the Cox model) is smaller than an independent
            exponential variable with scale factor
            `censoring_factor * mean_time`, where `mean_time`
            is the empirical mean of drawn event times.
            Defaults to False.
        seed: {None, int, Sequence[int], SeedSequence, BitGenerator, Generator},
            optional
            The seed for reproducibility. Defaults to None.
        standardize_features: bool, optional
            Whether to standardize features or not. Defaults to True.
        dtype : `{"float64", "float32"}`, default="float64"
            Type of the arrays used.
        """
        self.n_samples = n_samples
        self.ndim = ndim
        self.features_type: Final = features_type
        self.rng = np.random.default_rng(seed)
        self.prop_treated = prop_treated
        self.overlap = overlap
        self.cate = cate
        self.propensity = propensity
        self.cov_corr = cov_corr
        self.scale_t = scale_t
        self.shape_t = shape_t
        self.censoring_factor = censoring_factor
        self.random_censoring = random_censoring
        self.standardize_features = standardize_features
        self.dtype: Final = dtype
        self.coeffs = None
        self.percent_ties = percent_ties
        self.average_treatment_effect_ = None
        self.probability_treated = None

    def standardize_data(self, features: np.ndarray):
        """Standardize data. Make data reduced centered.

        Standardize the data by substracting the mean of each columns
        and dividing by the standard deviation.

        Parameters
        ----------
        features : np.ndarray
            Features to standardize.

        Returns
        -------
        np.ndarray
            Normalized features.
        """
        features -= features.mean(axis=0)
        features /= features.std(axis=0)
        return features

    def generate_data(
        self,
        n_samples: Optional[int] = None,
        seed: _SeedType = None,
        use_cate: bool = True,
    ):
        """Generate final survival data.

        Use the collection of methods of the class to
        generate data following Cox assumptions.

        Returns
        -------
        tuple
            A tuple of np.ndarrays.

        Raises
        ------
        ValueError
            If `propensity` is neither "constant" nor "linear".
        ValueError
            If `cate` is neither "linear", "random" nor a constant type int or float.
        """
        if n_samples is None:
            n_samples = self.n_samples
        if seed is None:
            seed = self.rng
        rng = np.random.default_rng(seed)

        if self.features_type == "cov_uniform":
            X = features_normal_cov_uniform(
                n_samples, self.ndim, dtype=self.dtype, seed=rng
            )
        elif self.features_type == "indep_gauss":
            X = rng.standard_normal(size=(n_samples, self.ndim)).astype(self.dtype)
        else:
            X = features_normal_cov_toeplitz(
                n_samples, self.ndim, self.cov_corr, dtype=self.dtype, seed=rng
            )
        if self.standardize_features:
            X = self.standardize_data(X)

        if self.propensity == "constant":
            treat_alloc = random_treatment_allocation(
                n_samples, self.prop_treated, seed=rng
            )
            propensity_scores = np.repeat(self.prop_treated, n_samples)

        elif self.propensity == "linear":
            func_propensity = linear_propensity(
                ndim=self.ndim,
                overlap=self.overlap,
                prop_treated=self.prop_treated,
                seed=rng,
            )
            propensity_scores = np.apply_along_axis(func_propensity, -1, X)
            treat_alloc = rng.binomial(1, propensity_scores)
        else:
            raise ValueError("propensity must be either `constant` or `linear`")

        self.coeffs = rng.normal(size=(self.ndim,)).astype(self.dtype)
        u = X.dot(self.coeffs)
        if use_cate:
            if self.cate == "linear":
                func_cate = linear_cate(ndim=self.ndim, seed=rng)
            elif self.cate == "random":
                func_cate = random_cate(seed=rng)
            elif isinstance(self.cate, (int, float)):
                func_cate = constant_cate(self.cate)
            else:
                raise ValueError(
                    """cate must be either `linear`, `random` or a constant type
                    int or float"""
                )

            cate_vector = np.apply_along_axis(func_cate, -1, X)
            self.average_treatment_effect_ = np.mean(cate_vector[treat_alloc == 1])
            self.probability_treated = cate_vector
            u += treat_alloc * np.log(cate_vector)
        # Simulation of true times
        time_hazard_baseline = -np.log(
            rng.uniform(0, 1.0, size=n_samples).astype(self.dtype)
        )
        time_cox_unscaled = time_hazard_baseline * np.exp(-u)
        times = self.scale_t * time_cox_unscaled ** (1.0 / self.shape_t)

        # induce samples with same times
        if self.percent_ties is not None:
            nb_ties_target = int(self.percent_ties * n_samples)
            if nb_ties_target >= 2:
                # sklearn not supporting generator yet, pass int to random_state
                # ref: https://github.com/scikit-learn/scikit-learn/issues/16988
                seed_seq = rng.bit_generator._seed_seq.spawn(1)[0]  # type: ignore
                random_state = seed_seq.generate_state(1)[0]
                original_times = copy.deepcopy(times)
                # We progressively reduce the number of bins until there are
                # only 2 bins starting with npoints - 1 bins
                reached = False
                for nbins in range(n_samples - 1, 1, -1):
                    discretizer = KBinsDiscretizer(
                        n_bins=nbins,
                        encode="ordinal",
                        strategy="quantile",
                        random_state=random_state,
                    )
                    times = discretizer.fit_transform(original_times.reshape((-1, 1)))
                    nb_ties_reached = n_samples - len(np.unique(times))
                    if (nb_ties_reached - nb_ties_target) >= 0:
                        reached = True
                        break
                if not reached:
                    raise ValueError("This should not happen, lower percent_ties")
                times = times.reshape((-1))

            else:
                raise ValueError("Choose a larger number of ties")

        avg_time = times.mean()

        # Simulation of the censoring times. times is returned in absolute value
        if self.random_censoring:
            censoring = rng.uniform(size=n_samples) < self.censoring_factor
            times[censoring] = [rng.uniform(0, t) for t in times[censoring].tolist()]
            censoring = censoring.astype("uint8")
        else:
            c_sampled = rng.exponential(
                scale=self.censoring_factor * avg_time, size=n_samples
            ).astype(self.dtype)

            censoring = (times > c_sampled).astype("uint8")
            times[censoring] = np.minimum(times, c_sampled)

        return X, times, censoring, treat_alloc, propensity_scores

    def generate_dataframe(
        self,
        n_samples: Optional[int] = None,
        prefix: str = "X_",
        duration_col: str = "time",
        event_col: str = "event",
        treated_col: str = "treatment",
        ps_col: str = "propensity_scores",
        seed: _SeedType = None,
    ):
        """Generate dataframe."""
        (
            covariates,
            times,
            censoring,
            treatments,
            propensity_scores,
        ) = self.generate_data(n_samples, seed=seed)
        data = pd.DataFrame(covariates).add_prefix(prefix)
        data[duration_col] = times
        data[event_col] = 1 - censoring
        data[treated_col] = treatments
        data[ps_col] = propensity_scores
        return data


def features_normal_cov_uniform(
    n_samples: int = 200,
    n_features: int = 30,
    dtype: Literal["float32", "float64"] = "float64",
    seed: _SeedType = None,
):
    """Generate Normal features with uniform covariance.

    An example of features obtained as samples of a centered Gaussian
    vector with a specific covariance matrix given by 0.5 * (U + U.T),
    where U is uniform on [0, 1] and diagonal filled by ones.

    Parameters
    ----------
    n_samples : int
        Number of samples. Default=200.
    n_features : int
        Number of features. Default=30.
    dtype : `{"float64", "float32"}`, optional
        Type of the arrays used. Default='float64'
    seed: {None, int, Sequence[int], SeedSequence, BitGenerator, Generator}, optional
        The seed for reproducibility. Defaults to None.

    Returns
    -------
    output : numpy.ndarray, shape=(n_samples, n_features)
        n_samples realization of a Gaussian vector with the described
        covariance
    """
    rng = np.random.default_rng(seed)
    pre_cov = rng.uniform(size=(n_features, n_features)).astype(dtype)
    np.fill_diagonal(pre_cov, 1.0)
    cov = 0.5 * (pre_cov + pre_cov.T)
    features = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    if dtype != "float64":
        return features.astype(dtype)
    return features


def features_normal_cov_toeplitz(
    n_samples: int = 200,
    n_features: int = 30,
    cov_corr: float = 0.5,
    dtype: Literal["float32", "float64"] = "float64",
    seed: _SeedType = None,
):
    """Generate normal features with toeplitz covariance.

    An example of features obtained as samples of a centered Gaussian vector with
    a toeplitz covariance matrix.

    Parameters
    ----------
    n_samples : int
        Number of samples. Default=200.
    n_features : int
        Number of features. Default=30.
    cov_corr : float
        correlation coefficient of the Toeplitz correlation matrix. Default=0.5.
    dtype : `{'float64', 'float32'}`, optional
        Type of the arrays used. Default='float64'
    seed: {None, int, Sequence[int], SeedSequence, BitGenerator, Generator}, optional
        The seed for reproducibility. Defaults to None.

    Returns
    -------
    output : numpy.ndarray, shape=(n_samples, n_features)
        n_samples realization of a Gaussian vector with the described
        covariance
    """
    rng = np.random.default_rng(seed)
    cov: np.ndarray = toeplitz(cov_corr ** np.arange(0, n_features))
    features = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    if dtype != "float64":
        return features.astype(dtype)
    return features


def make_categorical(X, up_to: int = 25, seed: _SeedType = None):
    """Convert continuous features in a dataset to categorical features.

    This function takes a dataset matrix `X` and converts its first `up_to` columns
    (features) into categorical features using the KBinsDiscretizer method.
    It performs min-max scaling on each feature before discretization.

    Parameters
    ----------
    X : np.ndarray
        Input dataset matrix of shape (n_samples, n_features).
    up_to : int, optional
        Number of columns to convert to categorical features, by default 25.
    seed : int or None, optional
        Seed for the random number generator, by default None.

    Returns
    -------
    np.ndarray, np.ndarray
        Two arrays: `Xleft` containing the modified categorical features
        and `Xright` containing the remaining original features.
    """
    rng = np.random.default_rng(seed)
    Xleft = X[:, :up_to]
    Xright = X[:, up_to:]
    mm_normalizer = MinMaxScaler()
    nbins_vector = rng.integers(2, 10, size=up_to)
    for j, nbins in enumerate(nbins_vector):
        # sklearn not supporting generator yet, pass int to random_state
        # ref: https://github.com/scikit-learn/scikit-learn/issues/16988
        seed_seq = rng.bit_generator._seed_seq.spawn(1)[0]  # type: ignore
        random_state = seed_seq.generate_state(1)[0]
        discretizer = KBinsDiscretizer(
            n_bins=nbins, encode="ordinal", random_state=random_state
        )
        Xleft[:, j] = mm_normalizer.fit_transform(Xleft[:, j][:, None])[:, 0]
        Xleft[:, j] = discretizer.fit_transform(Xleft[:, j][:, None])[:, 0]
    return Xleft, Xright


def generate_survival_data(
    n_samples: int = 100,
    ndim: int = 50,
    censoring_factor: float = 0.7,
    cate: float = 0.7,
    prop_treated: float = 0.5,
    ncategorical: int = 25,
    na_proportion: float = 0.1,
    dtype: Literal["float32", "float64"] = "float64",
    seed: _SeedType = None,
    use_cate: bool = True,
):
    """Generate simulated survival data.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples in the generated dataset, by default 100.
    ndim : int, optional
        Number of total features, by default 50.
    censoring_factor : float, optional
        Factor influencing the amount of censoring, by default 0.7.
    cate : float, optional
        CATE (Conditional Average Treatment Effect) parameter, by default 0.7.
    prop_treated : float, optional
        Proportion of the treated in the gobal population if treatments were to
        be assigned according to propensity scores generated by the returned
        function., by default 0.5.
    ncategorical : int, optional
        Number of categorical features, by default 25.
    na_proportion : float, optional
        Proportion of missing values, by default 0.1.
    dtype : Literal["float32", "float64"], optional
        Data type for the generated data, by default "float64".
    seed : int or None, optional
        Seed for the random number generator, by default None.
    use_cate : bool, optional
        Whether to use CATE (Conditional Average Treatment Effect), by default True.

    Returns
    -------
    pd.DataFrame, np.ndarray
        A pandas DataFrame containing the generated dataset with categorical and
        continuous features, and an array of coefficients used in the simulation.
    """
    assert ncategorical <= ndim
    rng = np.random.default_rng(seed)
    simu_coxreg = CoxData(
        n_samples,
        ndim=ndim,
        cate=cate,
        prop_treated=prop_treated,
        dtype=dtype,
        seed=rng,
        random_censoring=True,
        censoring_factor=censoring_factor,
        standardize_features=False,
    )
    X, T, C, treated, _ = simu_coxreg.generate_data(use_cate=use_cate)
    # Will make first columns to be categorical
    Xcat, Xcont = make_categorical(X, up_to=ncategorical)
    # Build the final dataframe using appropriate column names and adding missing
    # values
    cols_dict = {}
    X = np.concatenate((Xcat, Xcont), axis=1)
    for i in range(Xcat.shape[1] + Xcont.shape[1]):
        currentX = X[:, i].astype(dtype)
        mask_na = rng.uniform(0, 1, X.shape[0]) > (1.0 - na_proportion)
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

    return df, simu_coxreg.coeffs


def constant_cate(cate: float = 1.0) -> _FuncCateType:
    """Wrap a constant function indicating the hazard ratio."""
    return lambda _: cate


def linear_cate(ndim: int, seed: _SeedType = None) -> _FuncCateType:
    """Give the cate as the exponential of the linear combination of features.

    Coefficients of the linear combination is randomly generated.

    Parameters
    ----------
    ndim: int
        Number of features
    seed: {None, int, Sequence[int], SeedSequence, BitGenerator, Generator}, optional
        The seed for reproducibility. Defaults to None.

    Returns
    -------
    Callable:
        A function that takes a row (sample) of features as input and returns
        the cate (hazard ratio).
    """
    rng = np.random.default_rng(seed)
    params = rng.uniform(-1, 1, ndim) / np.sqrt(ndim)
    return lambda x: np.exp(np.sum(x * params))


def random_cate(seed: _SeedType = None) -> _FuncCateType:
    """Wrap a function giving random values of cate (hazard ratio)."""
    rng = np.random.default_rng(seed)
    return lambda _: np.exp(rng.normal(0, 0.4))


def constant_propensity(prop_treated: float = 0.5) -> _FuncPropensityType:
    """Wrap a constant function indicating the propensity scores."""
    return lambda _: prop_treated


def linear_propensity(
    ndim: int,
    overlap: float = 0,
    prop_treated: float = 0.5,
    seed: _SeedType = None,
) -> _FuncPropensityType:
    """Give the propensity scores as a linear combination of features.

    Coefficients of the linear combination is randomly generated.

    Parameters
    ----------
    ndim: int
        Number of features
    overlap: float, default=0
        Parameter controlling the strength of interaction between features and
        treatment allocation. The larger the strength, the weaker the overlap
        between the distributions of propensity scores of the treated group and
        the control group.
    prop_treated: float, default=0.5
        proportion of samples in the treated group if treatments were to be
        assigned according to propensity scores generated by the returned
        function.
    seed: {None, int, Sequence[int], SeedSequence, BitGenerator, Generator}, optional
        The seed for reproducibility. Defaults to None.

    Returns
    -------
    Callable:
        A function that takes a row (sample) of features as input and returns
        the probability of the sample being treated.
    """

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    rng = np.random.default_rng(seed)
    params = (1.0 + overlap) * rng.uniform(-0.5, 0.5, ndim) / np.sqrt(ndim)
    return lambda x: sigmoid(
        np.log(prop_treated / (1.0 - prop_treated)) + np.sum(x * params)
    )


def jacobian_torch(y, x, create_graph=False):
    """Compute the Jacobian of a vector-valued function with respect to its input.

    This function calculates the Jacobian matrix of a vector-valued function 'y'
    with respect to its input 'x', both represented as PyTorch tensors.
    The function computes the partial derivatives of each component of 'y' with
    respect to each element of 'x'.

    Parameters
    ----------
    y : torch.Tensor
        Output tensor of the vector-valued function, with shape (n_samples, n_outputs).
    x : torch.Tensor
        Input tensor with respect to which the Jacobian is computed, with shape
        (n_samples, n_inputs).
    create_graph : bool, optional
        If True, create a computation graph to allow further differentiation,
        by default False.

    Returns
    -------
    torch.Tensor
        Jacobian matrix of shape (n_samples, n_outputs, n_inputs), representing
        the derivatives of each component of 'y' with respect to each element of 'x'.
    """
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.0
        (grad_x,) = torch.autograd.grad(
            flat_y, x, grad_y, retain_graph=True, create_graph=create_graph
        )
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.0
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian_torch(y, x):
    """Compute the Hessian matrix of a scalar-valued function with respect to its input.

    This function calculates the Hessian matrix of a scalar-valued function 'y' with
    respect to its input 'x', both represented as PyTorch tensors.
    The Hessian matrix represents the second-order partial derivatives of the function
    'y' with respect to each pair of input elements.

    Parameters
    ----------
    y : torch.Tensor
        Output tensor of the scalar-valued function, with shape (n_samples,).
    x : torch.Tensor
        Input tensor with respect to which the Hessian is computed, with shape
        (n_samples, n_inputs).

    Returns
    -------
    torch.Tensor
        Hessian matrix of shape (n_samples, n_inputs, n_inputs), representing
        the second-order partial derivatives of the scalar-valued function 'y'
        with respect to each pair of input elements.
    """
    return jacobian_torch(jacobian_torch(y, x, create_graph=True), x)


def cox_partial_loglikelihood_breslow_torch(m, X, y):
    """Calculate the Cox partial log-likelihood using the Breslow estimator.

    This function computes the partial log-likelihood for the Cox Proportional Hazards
    model using the Breslow estimator. The partial log-likelihood quantifies the
    likelihood of observing the event times given the input features and
    model parameters.

    Parameters
    ----------
    m : CoxPHModelTorch
        An instance of the CoxPHModelTorch class representing the Cox Proportional
        Hazards model.
    X : np.ndarray or torch.Tensor
        Input feature matrix of shape (n_samples, n_features).
    y : np.ndarray or torch.Tensor
        Survival or event times, where positive values indicate observed events,
        and non-positive values indicate censored observations. Should have the
        same length as the number of samples.

    Returns
    -------
    float
        The negative of the Cox partial log-likelihood using the Breslow estimator.
    """
    # distinct event times
    distinct_times = np.unique(y[y > 0])
    Ds = []
    Rs = []
    for t in distinct_times:
        Ds.append(np.where(y == t)[0])
        Rs.append(np.where(np.abs(y) >= t)[0])
    coxl = 0.0
    beta = m.fc1.weight.permute((1, 0))
    for i, t in enumerate(distinct_times):
        XD = torch.from_numpy(X[Ds[i], :])
        XR = torch.from_numpy(X[Rs[i], :])

        if XD.shape[0] > 0:
            coxl += torch.dot(beta[:, 0], XD.sum(axis=0))  # type: ignore
        if XR.shape[0] > 0:
            expbetaX = m(XR)
            coxl -= float(len(Ds[i])) * torch.log(expbetaX.sum(axis=(0)))[0]

    return -coxl


def cox_partial_loglikelihood_breslow_torch_from_prediction(y_pred, y):
    """Compute the partial loglikelihood from the prediction y_pred of the model.

    This prediction corresponds to the score beta^T.Z of a model, before taking the
    exponential.

    Parameters
    ----------
    y_pred: torch.Tensor
        Prediction of the model, corresponding to the scores.
        The smaller the score is, the longer the survival.
    y: np.ndarray
        contains the labels of the data. Negative value corresponds to censored data
    Returns
    -------
        torch.Tensor: the negative log-lokelihood of Breslow-Cox
    """
    distinct_times = np.unique(y[y > 0])
    Ds = []
    Rs = []
    for t in distinct_times:
        Ds.append(np.where(y == t)[0])
        Rs.append(np.where(np.abs(y) >= t)[0])
    coxl = 0.0
    for i, t in enumerate(distinct_times):
        coxl += y_pred[Ds[i]].sum()
        if len(Rs[i]) > 0:
            expbetaX = torch.exp(y_pred[Rs[i]])
            coxl -= float(len(Ds[i])) * torch.log(expbetaX.sum(axis=0))  # type: ignore
    return -coxl


def analytical_gradient_cox_partial_loglikelihood_breslow_torch(m, X, y):
    """Calculate Cox partial log-likelihood's gradient using Breslow estimator.

    This function computes the analytical gradient of the partial log-likelihood
    for the Cox Proportional Hazards model using the Breslow estimator.
    The gradient is computed with respect to the model's weights.

    Parameters
    ----------
    m : CoxPHModelTorch
        An instance of the CoxPHModelTorch class representing the Cox
        Proportional Hazards model.
    X : np.ndarray or torch.Tensor
        Input feature matrix of shape (n_samples, n_features).
    y : np.ndarray or torch.Tensor
        Survival or event times, where positive values indicate observed events,
        and non-positive values indicate censored observations. Should have the same
        length as the number of samples.

    Returns
    -------
    torch.Tensor
        The analytical gradient of the partial log-likelihood with respect to
        the model's weights. The shape of the tensor matches the shape of the model's
        weight tensor.
    """
    # Obs = (y > 0).sum()

    # distinct event times
    with torch.no_grad():
        distinct_times = np.unique(y[y > 0])
        Ds = []
        Rs = []
        for t in distinct_times:
            Ds.append(np.where(y == t)[0])
            Rs.append(np.where(np.abs(y) >= t)[0])
        grad = torch.zeros_like(m.fc1.weight)

        for i, t in enumerate(distinct_times):
            XD = torch.from_numpy(X[Ds[i], :])
            XR = torch.from_numpy(X[Rs[i], :])
            expbetaX = m(XR)
            num = torch.mul(XR, expbetaX).sum(axis=0)  # type: ignore
            den = expbetaX.sum(axis=(0, 1))
            grad += XD.sum(axis=0) - float(  # type: ignore
                len(Ds[i])
            ) * num / torch.max(den, torch.ones_like(den) * 1e-16)

    return -grad


def random_treatment_allocation(
    n_samples: int,
    prop_treated: float = 0.5,
    shuffle: bool = False,
    seed: _SeedType = None,
) -> NDArray[np.uint8]:
    """Perform random treatment allocation for a given number of samples.

    This function generates a random allocation of treatments to samples
    based on the specified proportion of treated samples. The allocation can
    be optionally shuffled.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    prop_treated : float, optional
        Proportion of treated samples, by default 0.5.
    shuffle : bool, optional
        Whether to shuffle the treatment allocation, by default False.
    seed : int or None, optional
        Seed for the random number generator, by default None.

    Returns
    -------
    NDArray[np.uint8]
        An array of treatment allocations, where 1 represents treated and 0
        represents control. The array has a shape of (n_samples,).

    Examples
    --------
    >>> allocations = random_treatment_allocation(
    >>>    n_samples=100, prop_treated=0.3, shuffle=True
    >>> )
    >>> num_treated = np.sum(allocations)
    >>> num_control = n_samples - num_treated
    >>> print(f"Number of treated samples: {num_treated}")
    >>> print(f"Number of control samples: {num_control}")
    """
    rng = np.random.default_rng(seed)

    prop_control = 1 - prop_treated
    n_treated = int(n_samples * prop_treated)
    n_control = int(n_samples * prop_control)
    n_remained = n_samples - n_treated - n_control  # 1 or 0

    allocations = np.concatenate(
        (
            np.repeat(1, n_treated),
            np.repeat(0, n_control),
            np.repeat(
                rng.choice([1, 0], size=1, p=[prop_treated, prop_control]),
                n_remained,
            ),
        )
    ).astype("uint8")
    if shuffle:
        rng.shuffle(allocations)

    return allocations


class CoxPHModelTorch(torch.nn.Module):
    """Cox Proportional Hazards Model implemented using PyTorch.

    This class defines a Cox Proportional Hazards model as a PyTorch Module.
    The model takes input features and predicts the exponentiated linear term.
    The exponentiated linear term can be interpreted as the hazard ratio.

    Parameters
    ----------
    ndim : int, optional
        Number of input dimensions or features, by default 10.
    torch_dtype : torch.dtype, optional
        Data type for PyTorch tensors, by default torch.float64.
    """

    def __init__(self, ndim=10, torch_dtype=torch.float64):
        """Initialize the CoxPHModelTorch.

        Parameters
        ----------
        ndim : int, optional
            Number of input dimensions or features, by default 10.
        torch_dtype : torch.dtype, optional
            Data type for PyTorch tensors, by default torch.float64.
        """
        super().__init__()
        self.ndim = ndim
        self.torch_dtype = torch_dtype
        self.fc1 = torch.nn.Linear(self.ndim, 1, bias=False).to(self.torch_dtype)
        self.fc1.weight.data.fill_(0.0)

    def forward(self, x):
        """Perform a forward pass through the CoxPH model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ndim).

        Returns
        -------
        torch.Tensor
            Predicted exponentiated linear term (hazard ratio).
        """
        return torch.exp(self.fc1(x))  # pylint: disable=not-callable


class MockStepSizer:
    """A mock step sizer class for illustrative purposes.

    This class represents a simple mock step sizer that doesn't perform any actual step
    size calculation. It always returns a constant step size of 1.0 when the `next()`
    method is called.
    """

    def __init__(self):
        """Init method for the class."""
        pass

    def update(self, *args, **kwargs):
        """Update the state of the mock step sizer.

        Parameters
        ----------
        *args, **kwargs : arguments and keyword arguments
            Additional arguments that might be used to update the state (ignored
            in this mock implementation).

        Returns
        -------
        self : MockStepSizer
            Returns the instance of the MockStepSizer class itself.
        """
        return self

    def next(self):
        """Get the next step size.

        Returns
        -------
        float
            The constant step size of 1.0.
        """
        return 1.0


def compute_q_k(
    X_norm,
    y,
    scaled_variance_matrix,
    distinct_event_times,
    weights_counts_on_events,
    risk_phi,
    risk_phi_x,
    score,
    weights,
):
    """Compute local bricks for Q.

    Parameters
    ----------
    X_norm : np.ndarray
        _description_
    y : np.ndarray
        _description_
    scaled_variance_matrix : np.ndarray
        _description_
    distinct_event_times : list[float]
        _description_
    weights_counts_on_events : list[float]
        _description_
    risk_phi : list[float]
        _description_
    risk_phi_x : list[np.ndarray]
        _description_
    score : np.ndarray
        _description_
    weights : np.ndarray
        _description_

    Returns
    -------
    tuple(np.ndarray, np.ndarray, np.ndarray)
        _description_
    """
    n, n_features = X_norm.shape
    phi_k = np.zeros((n, n_features))
    current_client_indices = np.arange(n).tolist()
    weights_counts_on_events_cumsum = np.concatenate(
        [wc.reshape((1, 1)) for wc in weights_counts_on_events],
        axis=0,
    )
    s0s_cumsum = np.concatenate(
        [risk_phi_s.reshape((1, 1)) for risk_phi_s in risk_phi],
        axis=0,
    )
    s1s_cumsum = np.concatenate(
        [risk_phi_x_s.reshape((1, n_features)) for risk_phi_x_s in risk_phi_x],
        axis=0,
    )
    # # of size (i + 1, n_features) this should be term by term
    s1_over_s0_cumsum = s1s_cumsum / (s0s_cumsum)

    # The division should happen term by term
    weights_over_s0_cumsum = weights_counts_on_events_cumsum / s0s_cumsum

    for i in current_client_indices:
        # This is the crux of the implementation, we only have to sum on times
        # with events <= ti
        # as otherwise delta_j = 0 and therefore the term don't contribute
        ti = np.abs(y[i])

        compatible_event_times = [
            idx for idx, td in enumerate(distinct_event_times) if td <= ti
        ]
        # It can happen that we have a censorship that happens before any
        # event times
        if len(compatible_event_times) > 0:
            # distinct_event_times is sorted so we can do that
            max_distinct_event_times = max(compatible_event_times)
            # These are the only indices of the sum, which will be active
            not_Rs_i = np.arange(max_distinct_event_times + 1)

            # Quantities below are global and used onl alread shared quantities
            s1_over_s0_in_sum = s1_over_s0_cumsum[not_Rs_i]
            weights_over_s0_in_sum = weights_over_s0_cumsum[not_Rs_i]

        else:
            # There is nothing in the sum we'll add nothing
            s1_over_s0_in_sum = np.zeros((n_features))
            weights_over_s0_in_sum = 0.0

        # Second term and 3rd term
        phi_i = -score[i] * (
            weights_over_s0_in_sum * (X_norm[i, :] - s1_over_s0_in_sum)
        ).sum(axis=0).reshape((n_features))

        # First term
        if y[i] > 0:
            phi_i += (
                X_norm[i, :]
                - risk_phi_x[max_distinct_event_times][None, :]
                / risk_phi[max_distinct_event_times][None, None]
            ).reshape((n_features))

        # We recallibrate by w_i only at the very end in here we deviate a
        # bit from Binder ?

        phi_k[i] = phi_i * weights[i]

    # We have computed scaled_variance_matrix globally this delta_beta is
    # (n_k, n_features)
    delta_betas_k = phi_k.dot(scaled_variance_matrix)
    # This Qk is n_features * n_features we will compute Q by block
    Qk = delta_betas_k.T.dot(delta_betas_k)

    return phi_k, delta_betas_k, Qk


def robust_sandwich_variance_distributed(
    X_norm, y, scaled_beta, weights, scaled_variance_matrix, n_clients=3
):
    """Compute the robust sandwich variance estimator.

    This function computes the robust sandwich variance estimator for the Cox
    model. The sandwich variance estimator is a robust estimator of the variance
    which accounts for the lack of dependence between the samples due to the
    introduction of weights for example.

    X_norm : np.ndarray or torch.Tensor
        Input feature matrix of shape (n_samples, n_features).
    y : np.ndarray or torch.Tensor
        Survival or event times, where positive values indicate observed events,
        and non-positive values indicate censored observations. Should have the
        same length as the number of samples.
    scaled_beta : np.ndarray or torch.Tensor
        The model's coefficients, with shape (n_features,).
    weights : np.ndarray or torch.Tensor
        Weights associated with each sample, with shape (n_samples,)
    scaled_variance_matrix : np.ndarray or torch.Tensor
        Classical scaled variance of the Cox model estimator.
    """
    n_samples = X_norm.shape[0]

    np.random.seed(42)
    samples_repartition = np.random.choice(n_clients, size=n_samples)

    # This part is already computed for WebDisco
    # in fact we have the D_i and R_i by client and share global distinct_times

    score = np.exp(np.dot(X_norm, scaled_beta))

    # np.unique already sorts values to need to call sort another time
    distinct_event_times = np.unique(np.abs(y)).tolist()

    Ds = []
    Rs = []
    for t in distinct_event_times:
        Ds.append(np.where(y == t)[0])
        Rs.append(np.where(np.abs(y) >= t)[0])

    # This is assumed to be globally available
    # for risk_phi each element is a scalar
    risk_phi = []
    # for risk_phi_x each element is of the dimension of a feature N,
    risk_phi_x = []
    weights_counts_on_events = []

    for i, _ in enumerate(distinct_event_times):
        risk_phi_x.append(
            np.sum(X_norm[Rs[i], :] * (weights[Rs[i]] * score[Rs[i]])[:, None], axis=0)
        )
        risk_phi.append(np.sum((weights[Rs[i]] * score[Rs[i]])[:, None]))
        weights_counts_on_events.append(weights[Ds[i]].sum())
    # Iterate forwards
    Q = []

    for k in range(n_clients):
        indices_client_k = np.where(samples_repartition == k)
        X_norm_k = X_norm[indices_client_k]
        y_k = y[indices_client_k]
        weights_k = weights[indices_client_k]
        score_k = score[indices_client_k]
        _, _, Qk = compute_q_k(
            X_norm_k,
            y_k,
            scaled_variance_matrix,
            distinct_event_times,
            weights_counts_on_events,
            risk_phi,
            risk_phi_x,
            score_k,
            weights_k,
        )
        # Communication to the server
        Q.append(Qk)

    # We sum each block
    Q = sum(Q)
    return np.sqrt(np.diag(Q))


def robust_sandwich_variance_pooled(
    X_norm, y, scaled_beta, weights, scaled_variance_matrix
):
    """Compute the robust sandwich variance estimator.

    This function computes the robust sandwich variance estimator for the Cox
    model. The sandwich variance estimator is a robust estimator of the variance
    which accounts for the lack of dependence between the samples due to the
    introduction of weights for example.
    X_norm : np.ndarray or torch.Tensor
        Input feature matrix of shape (n_samples, n_features).
    y : np.ndarray or torch.Tensor
        Survival or event times, where positive values indicate observed events,
        and non-positive values indicate censored observations. Should have the
        same length as the number of samples.
    scaled_beta : np.ndarray or torch.Tensor
        The model's coefficients, with shape (n_features,).
    weights : np.ndarray or torch.Tensor
        Weights associated with each sample, with shape (n_samples,)
    scaled_variance_matrix : np.ndarray or torch.Tensor
        Classical scaled variance of the Cox model estimator.
    """
    n_samples, n_features = X_norm.shape

    score_residuals = np.zeros((n_samples, n_features))

    phi_s = np.exp(np.dot(X_norm, scaled_beta))

    distinct_times = sorted(np.unique(np.abs(y)))
    Ds = []
    Rs = []
    for t in distinct_times:
        Ds.append(np.where(np.abs(y) == t)[0])
        Rs.append(np.where(np.abs(y) >= t)[0])

    risk_phi_x_history = np.zeros((n_samples, n_features))
    risk_phi_history = np.zeros(n_samples)
    for i, t in enumerate(distinct_times):
        for j in Ds[i]:
            risk_phi_x_history[j, :] = np.sum(
                X_norm[Rs[i], :] * (weights[Rs[i]] * phi_s[Rs[i]])[:, None], axis=0
            )
            risk_phi_history[Ds[i]] = np.sum((weights[Rs[i]] * phi_s[Rs[i]])[:, None])
    # Iterate forwards
    for i, t in enumerate(distinct_times):
        for j in Ds[i]:
            not_Rs = set(np.arange(n_samples)) - set(Rs[i])
            not_Rs = list((not_Rs.union(np.array([j]))))
            score_residuals[j, :] = -phi_s[j] * (
                ((y > 0)[not_Rs] * weights[not_Rs] / risk_phi_history[not_Rs])[:, None]
                * (
                    X_norm[j, :]
                    - risk_phi_x_history[not_Rs] / risk_phi_history[not_Rs][:, None]
                )
            ).sum(axis=0)

            if y[j] > 0:
                score_residuals[j, :] += (
                    X_norm[j, :] - risk_phi_x_history[j] / risk_phi_history[j]
                )

    score_residuals = score_residuals * weights[:, None]

    delta_betas = score_residuals.dot(scaled_variance_matrix)
    tested_var = delta_betas.T.dot(delta_betas)
    return np.sqrt(np.diag(tested_var))


def km_curve(t, n, d, tmax=5000):
    """Computes Kaplan-Meier (KM) curve based on unique event times, number of
    individuals at risk and number of deaths.

    This function is typically used in conjunction with
    `compute_events_statistics`. Note that the variance is computed
    based on Greenwood's formula, not its exponential variant (see refs).

    Parameters
    ----------
    t : np.array
        Array containing the unique times of events, sorted in ascending order
    n : np.array
        Array containing the number of individuals at risk at each
        corresponding time `t`
    d : np.array
        Array containing the number of individuals with an event (death) at
        each corresponding time `t`
    tmax : int, optional
        Maximal time point for the KM curve, by default 5000

    Returns
    -------
    tuple
        Tuple of length 3 containing:
        - `grid`: 1D array containing the time points at which the survival
          curve is evaluated. It is `np.arange(0, t_max+1)`
        - `s`: 1D array with the value of the survival function as obtained
          by the Kaplan-Meier formula.
        - `var_s`: `np.array` containing the variance of the Kaplan-Meier curve
          at each point of `grid`

    Examples
    --------
    .. code-block:: python
       :linenos:
       # Define times and events
       times = np.random.randint(0, 3000, size=(10,))
       events = np.random.rand(10) > 0.5
       # Compute events statistics
       t, n, d = compute_events_statistics(times, events)
       # Get Kaplan-Meier curve
       grid, s, var_s = km_curve(t, n, d)

       # Plot KM curve
       plt.figure()
       plt.plot(grid, s)
       plt.fill_between(grid, s-np.sqrt(var_s), s+np.sqrt(var_s))

    References
    ----------
    https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator
    https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf
    """
    grid = np.arange(0, tmax + 1)
    # precompute for KM
    q = 1.0 - d / n
    cprod_q = np.cumprod(q)
    # precompute for var
    csum_var = np.cumsum(d / (n * (n - d)))
    # initialize
    s = np.zeros(grid.shape)
    var_s = np.zeros(grid.shape)
    # the following sum relies on the assumption that unique_events_times is sorted!
    index_closest = np.sum(grid.reshape(-1, 1) - t.reshape(1, -1) >= 0, axis=1) - 1
    not_found = index_closest < 0
    found = index_closest >= 0
    # attribute
    s[not_found] = 1.0
    s[found] = cprod_q[index_closest[found]]
    var_s[found] = (s[found] ** 2) * csum_var[index_closest[found]]
    return grid, s, var_s


def compute_events_statistics(times, events):
    """Compute unique times, number of individuals at risk at these times and number of
    events at these times based on the raw list of individual times and events, for a
    survival framework.

    The method is vectorized with numpy to ensure fast computations. As a
    side-effect, memory consumption scales as
    `num_unique_times * num_individuals`.

    Parameters
    ----------
    times : np.array
        1D array containing the individual observed times, which are either
        censored times or true event times depending on the corresponding value
        of the `events` array
    events : np.array
        1D array with boolean entries, such that `events[i] == True` if and
        only if a true event was observed for individual `i` at time `times[i]`

    Returns
    -------
    tuple
        Tuple of length 3 containing, in this order:
        - `unique_times`: `np.array` containing the unique times of events,
          in ascending order
        - `num_at_risk_at_times`: `np.array` containing the number of individuals
          at risk at each corresponding `unique_times`
        - `num_death_at_times`: `np.array` containing the number of individuals
          with an event (death) at each corresponding `unique_times`

    Examples
    --------
    .. code-block:: python
       :linenos:
       # Define times and events
       times = np.random.randint(0, 3000, size=(10,))
       events = np.random.rand(10) > 0.5
       # Compute events statistics
       t, n, d = compute_events_statistics(times, events)
    """
    # NB, both censored and uncensored, otherwise impossible to aggregate exactly
    unique_times = np.unique(times)
    num_death_at_times = np.sum(
        (unique_times.reshape(-1, 1) - times[events].reshape(1, -1)) == 0, axis=1
    )
    num_at_risk_at_times = np.sum(
        (times.reshape(1, -1) - unique_times.reshape(-1, 1)) >= 0, axis=1
    )
    return unique_times, num_at_risk_at_times, num_death_at_times


def aggregate_events_statistics(list_t_n_d):
    """Aggregates (sums) events statistics from different centers, returning a single
    tuple with the same format.

    Parameters
    ----------
    list_t_n_d : tuple
        List of event statistics from different centers. Each entry in the list
        should follow the output format of `compute_events_statistics`

    Returns
    -------
    tuple
        Tuple of size 3 containing, in this order:
        - `t_agg`: `np.array` containing the unique times of events,
          in ascending order
        - `n_agg`: `np.array` containing the number of individuals
          at risk at each corresponding `t_agg`
        - `d_agg`: `np.array` containing the number of individuals
          with an event (death) at each corresponding `t_agg`
    """
    # Step 1: get the unique times
    unique_times_list = [t for (t, n, d) in list_t_n_d]
    t_agg = np.unique(np.concatenate(unique_times_list))
    # Step 2: extend to common grid
    n_ext, d_ext = extend_events_to_common_grid(list_t_n_d, t_agg)
    # Step 3:sum across centers
    n_agg = np.sum(n_ext, axis=0)
    d_agg = np.sum(d_ext, axis=0)
    return t_agg, n_agg, d_agg


def extend_events_to_common_grid(list_t_n_d, t_common):
    """Extends a list of heterogeneous times, number of people at risk and number of
    death on a common grid.

    This method is an internal utility for `aggregate_events_statistics`.

    Parameters
    ----------
    list_t_n_d : List[tuple]
        List of tuples, each item in the list being an output of
        `compute_events_statistics`, i.e. each tuple is of size 3 and contains
        - `t`: 1D array with unique event times, sorted in ascending order
        - `n`: 1D array with the number of individuals at risk for each time point
        - `d`: 1D array with the number of events for each time point
    t_common : np.ndarray
        Common grid on which to compute the aggregated times.

    Returns
    -------
    tuple
        Tuple with 2 values, each being a 2D array of size
        `len(list_t_n_d) * t_common.size`
        - `n_ext`: number of individuals at risk for each center and time
          point in `t_common`
        - `d_ext`: number of events for each center and each time point
          in `t_common`
    """
    num_clients = len(list_t_n_d)

    d_ext = np.zeros((num_clients, t_common.size))
    n_ext = np.zeros((num_clients, t_common.size))

    for k, (t, n, d) in enumerate(list_t_n_d):
        diff = t - t_common.reshape(-1, 1)
        diff_abs = np.abs(diff)
        # identify times which were in the client vs those not
        is_in_subset = diff_abs.min(axis=1) == 0
        not_in_subset = diff_abs.min(axis=1) > 0
        # make a correspondence for those that were in
        corr_in_subset = np.zeros(t_common.size, dtype=int)
        corr_in_subset[is_in_subset] = diff_abs[is_in_subset].argmin(axis=1)
        # for those that were not, but which are inside the grid, we can extend n
        diff_relu = np.maximum(diff, 0)
        has_match = (diff_relu.max(axis=1) > 0) & not_in_subset
        # we rely on the fact that t is ordered!
        corr_in_subset[has_match] = np.sum(diff_relu == 0, axis=1)[has_match]
        # Get the deaths: 0 if no time, the value if it was inside
        d_ext[k, is_in_subset] = d[corr_in_subset[is_in_subset]]
        # Get the people at risk for those in subset
        n_ext[k, is_in_subset] = n[corr_in_subset[is_in_subset]]
        # For those not in subset but with a match, extend in a piecewise constant fashion
        n_ext[k, has_match] = n[corr_in_subset[has_match]]
    return n_ext, d_ext


def build_X_y_function(
    data_from_opener,
    event_col,
    duration_col,
    treated_col,
    target_cols=None,
    standardize_data=True,
    propensity_model=None,
    cox_fit_cols=None,
    propensity_fit_cols=None,
    tol=1e-16,
    training_strategy="webdisco",
    shared_state={},
    global_moments={},
):
    """Build the inputs for a propensity model and for a Cox model as well as y directly
    from the output of the opener.

    This function 1. uses the event column to inject the censorship
    information present in the duration column (given in absolute values)
    in the form of a negative sign producing y.
    2. Drops the raw survival columns (given in target_cols) now that y is built.
    2. Drops some covariates differently for Cox and for the propensity model
    according to the training_strategy argument as well as cox and propensity
    fit cols.
    4. Standardize the data if standardize_data. It will use either the shared_state
    or the global moments variable depending on what is given. If both given
    uses shared_state. If not given does not err but does not standardize.
    5. Return the Cox model input X as well as the (unstandardized) input to the
    propensity model Xprop if necessary as well as the treated column to be able
    to compute the propensity weights.

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
        2.0 * e - 1.0 for e in data_from_opener[event_col].tolist()
    ]
    # No funny business irrespective of the convention used
    y = np.abs(data_from_opener[duration_col]) * data_from_opener["time_multiplier"]
    y = y.to_numpy().astype("float64")
    data_from_opener.drop(columns=["time_multiplier"], inplace=True)
    # dangerous but we need to do it
    string_columns = [
        col
        for col in data_from_opener.columns
        if not (is_numeric_dtype(data_from_opener[col]))
    ]
    data_from_opener.drop(columns=string_columns, inplace=True)

    # We drop the targets from X
    if target_cols is None:
        target_cols = [event_col, duration_col]
    columns_to_drop = target_cols
    X = data_from_opener.drop(columns=columns_to_drop)
    if propensity_model is not None:
        assert treated_col is not None
        if training_strategy == "iptw":
            X = X.loc[:, [treated_col]]
        elif training_strategy == "aiptw":
            if len(cox_fit_cols) > 0:
                X = X.loc[:, [treated_col] + cox_fit_cols]
            else:
                pass
    else:
        assert training_strategy == "webdisco"
        if len(cox_fit_cols) > 0:
            X = X.loc[:, cox_fit_cols]
        else:
            pass

    # If X is to be standardized we do it
    if standardize_data:
        if shared_state:
            # Careful this shouldn't happen apart from the predict
            means = shared_state["global_uncentered_moment_1"]
            vars = shared_state["global_centered_moment_2"]
            # Careful we need to match pandas and use unbiased estimator
            bias_correction = (shared_state["total_n_samples"]) / float(
                shared_state["total_n_samples"] - 1
            )
            global_moments = {
                "means": means,
                "vars": vars,
                "bias_correction": bias_correction,
            }
            stds = vars.transform(lambda x: sqrt(x * bias_correction + tol))
            X = X.sub(means)
            X = X.div(stds)
        else:
            X = X.sub(global_moments["means"])
            stds = global_moments["vars"].transform(
                lambda x: sqrt(x * global_moments["bias_correction"] + tol)
            )
            X = X.div(stds)

    X = X.to_numpy().astype("float64")

    # If we have a propensity model we need to build X without the targets AND the
    # treated column
    if propensity_model is not None:
        # We do not normalize the data for the propensity model !!!
        Xprop = data_from_opener.drop(columns=columns_to_drop + [treated_col])
        if propensity_fit_cols is not None:
            Xprop = Xprop[propensity_fit_cols]
        Xprop = Xprop.to_numpy().astype("float64")
    else:
        Xprop = None

    # If WebDisco is used without propensity treated column does not exist
    if treated_col is not None:
        treated = (
            data_from_opener[treated_col].to_numpy().astype("float64").reshape((-1, 1))
        )
    else:
        treated = None

    return (X, y, treated, Xprop, global_moments)


def compute_X_y_and_propensity_weights_function(
    X, y, treated, Xprop, propensity_model, tol=1e-16
):
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

    if propensity_model is not None:
        assert (
            treated is not None
        ), f"""If you are using a propensity model the Treated
        column should be available"""
        assert np.all(
            np.in1d(np.unique(treated.astype("uint8"))[0], [0, 1])
        ), "The treated column should have all its values in set([0, 1])"
        Xprop = torch.from_numpy(Xprop)

        with torch.no_grad():
            propensity_scores = propensity_model(Xprop)

        propensity_scores = propensity_scores.detach().numpy()
        # We robustify the division
        weights = treated * 1.0 / np.maximum(propensity_scores, tol) + (
            1 - treated
        ) * 1.0 / (np.maximum(1.0 - propensity_scores, tol))
    else:
        weights = np.ones((X.shape[0], 1))
    return X, y, weights
