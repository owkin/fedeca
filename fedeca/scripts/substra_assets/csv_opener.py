"""CSV opener for substra."""
import pathlib

import numpy as np
import pandas as pd
import substratools as tools
from scipy.linalg.special_matrices import toeplitz
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler


class CSVOpener(tools.Opener):
    """CSV opener class."""

    def fake_data(self, n_samples=None):
        """Generate simulated survival data.

        Parameters
        ----------
        n_samples : int or None, optional
            Number of samples, by default None.

        Returns
        -------
        pd.DataFrame
            Fake survival data.
        """
        N_SAMPLES = n_samples if n_samples and n_samples <= 100 else 100
        return generate_survival_data(
            nsample=N_SAMPLES,
            na_proportion=0.0,
            ndim=10,
            seed=np.random.randint(0, 10000),
        )

    def get_data(self, folders):
        """Get data from CSV files.

        Parameters
        ----------
        folders : list
            List of folder paths.

        Returns
        -------
        pd.DataFrame
            Loaded data from CSV files.
        """
        # get npy files
        p = pathlib.Path(folders[0])
        csv_data_path = p / "data.csv"
        # load data
        data = pd.read_csv(csv_data_path)
        return data


class CoxData:
    """Simulate Cox data.

    This class simulates survival data following Cox model assumptions.
    """

    def __init__(
        self,
        n_samples=1000,
        ndim=10,
        features_type="cov_toeplitz",
        cov_corr=0.5,
        scale_t=0.1,
        shape_t=3.0,
        censoring_factor=0.5,
        random_censoring=False,
        seed=42,
        standardize_features=True,
        dtype="float64",
    ):
        r"""Cox Data generator class.

        This class generates data according to a Cox proportional hazards model
        in continuous time as follows:
        .. math::
          S(t|x) = P(T > t | X=x)
          \\lambda(t|x) = \\frac{d \\log S(t|x)}{dt}
          \\lambda(t|x) = \\lambda_0(t)e^{\\beta^T x}
          \\Lambda_0(t|x) = \\int_0^t \\lambda_0(u)du = (s \\times t)^k
          X \\sim \\mathcal{N}(0, C)
          \\beta \\sim \\mathcal{N}(0, I)

        Parameters
        ----------
        n_samples: int, optional
            Number of samples to generate. Defaults to 1000
        ndim: int, optional
            Number of features, defaults to 10.
        features_type: str, optional
            Accepted values: `"cov_toeplitz"`, `"cov_uniform"`.
        cov_corr: float, optional
            The correlation of the covariance matrix.
        scale_t: float, optional
            Scale parameter `s` in the equations above. Defaults to `1`.
        shape_t: float, optional
            Shape parameter `k` in the equations above. Defaults to `1`.
        censoring_factor: float, optional
            Parameter used to determine the probability of being censored
            (with respect to the median). Defaults to `0.5`.
        random_censoring: bool, optional
            Whether to censor completely independently of the rest or not.
            When true, censors samples with probability censoring_factor.
            When false, samples are censored if the drawn event times
            (drawn from the Cox model) is smaller than an independent
            exponential variable with scale factor
            `censoring_factor * mean_time`, where `mean_time`
            is the empirical mean of drawn event times.
            Defaults to False.
        seed: int, otional
            The seed for reproducibility.
        standardize_features: bool, optional
            Whether to standardize features or not. Defaults to True.
        dtype : `{'float64', 'float32'}`, default='float64'
            Type of the arrays used.
        """
        self.n_samples = n_samples
        self.ndim = ndim
        self.features_type = features_type
        self.cov_corr = cov_corr
        self.scale = scale_t
        self.shape = shape_t
        self.censoring_factor = censoring_factor
        self.random_censoring = random_censoring
        self.standardize_features = standardize_features
        self.dtype = dtype
        self.coeffs = None
        np.random.seed(seed)

    def standardize_data(self, features: np.ndarray):
        """Standardize data. Make data reduced centered.

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

    def generate_data(self):
        """Generate final survival data.

        Use the collection of methods of the class to
        generate data following Cox assumptions.

        Returns
        -------
        tuple
            A tuple of np.ndarrays.
        """
        if self.features_type == "cov_uniform":
            X = features_normal_cov_uniform(self.n_samples, self.ndim, dtype=self.dtype)
        elif self.features_type == "indep_gauss":
            X = np.random.randn(self.n_samples, self.ndim).astype(self.dtype)
        else:
            X = features_normal_cov_toeplitz(
                self.n_samples, self.ndim, self.cov_corr, dtype=self.dtype
            )
        if self.standardize_features:
            X = self.standardize_data(X)

        self.coeffs = np.random.normal(size=(self.ndim,)).astype(self.dtype)
        u = X.dot(self.coeffs)
        # Simulation of true times
        time_hazard_baseline = -np.log(
            np.random.uniform(0, 1.0, size=self.n_samples).astype(self.dtype)
        )
        time_cox_unscaled = time_hazard_baseline * np.exp(-u)
        times = 1.0 / self.scale * time_cox_unscaled ** (1.0 / self.shape)
        avg_time = times.mean()
        # Simulation of the censoring
        if self.random_censoring:
            censoring = np.random.rand(self.n_samples) < self.censoring_factor
            times[censoring] = [
                -np.random.uniform(0, t) for t in times[censoring].tolist()
            ]
            censoring = censoring.astype("uint8")
        else:
            c = self.censoring_factor
            c_sampled = np.random.exponential(
                scale=c * avg_time, size=self.n_samples
            ).astype(self.dtype)
            censoring = (times <= c_sampled).astype("uint8")
            times[censoring] = [
                -np.random.uniform(0, t) for t in times[censoring].tolist()
            ]
        return X, times, censoring


def features_normal_cov_uniform(
    n_samples: int = 200, n_features: int = 30, dtype="float64"
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
    dtype : str
        `{'float64', 'float32'}`,
        Type of the arrays used. Default='float64'
    Returns
    -------
    output : numpy.ndarray, shape=(n_samples, n_features)
        n_samples realization of a Gaussian vector with the described
        covariance
    """
    pre_cov = np.random.uniform(size=(n_features, n_features)).astype(dtype)
    np.fill_diagonal(pre_cov, 1.0)
    cov = 0.5 * (pre_cov + pre_cov.T)
    features = np.random.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    if dtype != "float64":
        return features.astype(dtype)
    return features


def features_normal_cov_toeplitz(
    n_samples: int = 200, n_features: int = 30, cov_corr: float = 0.5, dtype="float64"
):
    """Generate normal features with toeplitz covariance.

    An example of features obtained as samples of a centered Gaussian
    vector with a toeplitz covariance matrix.

    Parameters
    ----------
    n_samples : int
        Number of samples. Default=200.
    n_features : int
        Number of features. Default=30.
    cov_corr : float
        correlation coefficient of the Toeplitz correlation matrix. Default=0.5.
    dtype : str
        `{'float64', 'float32'}`,
        Type of the arrays used. Default='float64'
    Returns
    -------
    output : numpy.ndarray, shape=(n_samples, n_features)
        n_samples realization of a Gaussian vector with the described
        covariance
    """
    cov = toeplitz(cov_corr ** np.arange(0, n_features))
    features = np.random.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    if dtype != "float64":
        return features.astype(dtype)
    return features


def make_categorical(X, up_to=25):
    """Create categorical data.

    Parameters
    ----------
    X: nd.array
        Matrix from which to build the categorical features.
    up_to: int
        Takes up_to first columns to transform them into
        categorical data.
    """
    Xleft = X[:, :up_to]
    Xright = X[:, up_to:]
    mm_normalizer = MinMaxScaler()
    nbins_vector = np.random.randint(2, 10, size=up_to)
    for j, nbins in enumerate(nbins_vector):
        discretizer = KBinsDiscretizer(n_bins=nbins, encode="ordinal")
        Xleft[:, j] = mm_normalizer.fit_transform(Xleft[:, j][:, None])[:, 0]
        Xleft[:, j] = discretizer.fit_transform(Xleft[:, j][:, None])[:, 0]
    return Xleft, Xright


def generate_survival_data(
    n_samples=100,
    ndim=50,
    censoring_factor=0.7,
    seed=42,
    ncategorical=25,
    na_proportion=0.1,
    dtype="float32",
):
    """Generate synthetic survival data.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples, by default 100.
    ndim : int, optional
        Number of dimensions (features), by default 50.
    censoring_factor : float, optional
        Factor controlling censoring rate, by default 0.7.
    seed : int, optional
        Seed for random number generation, by default 42.
    ncategorical : int, optional
        Number of categorical features, by default 25.
    na_proportion : float, optional
        Proportion of missing values, by default 0.1.
    dtype : str, optional
        Data type for the generated data, by default "float32".

    Returns
    -------
    pd.DataFrame
        Synthetic survival data.
    """
    assert ncategorical <= ndim
    simu_coxreg = CoxData(
        n_samples,
        ndim=ndim,
        dtype=dtype,
        seed=seed,
        random_censoring=True,
        censoring_factor=censoring_factor,
        standardize_features=False,
    )
    X, T, C = simu_coxreg.generate_data()
    # Will make first columns to be categorical
    Xcat, Xcont = make_categorical(X, up_to=ndim // 2)
    # Build the final dataframe using appropriate column names and adding missing values
    cols_dict = {}
    X = np.concatenate((Xcat, Xcont), axis=1)
    for i in range(Xcat.shape[1] + Xcont.shape[1]):
        currentX = X[:, i].astype("float32")
        mask_na = np.random.uniform(0, 1, X.shape[0]) > (1.0 - na_proportion)
        currentX[mask_na] = np.nan
        if i < Xcat.shape[1]:
            colname = "cat_col"
        else:
            colname = "col"
            i -= Xcat.shape[1]
        cols_dict[f"{colname}_{i}"] = currentX
    # T is multiplied by -1 if censored and continuous,
    #  we make it so that the first time is 10 and other times are integers
    cols_dict["T"] = np.array(np.abs(T) - np.abs(T).min() + 10).astype("int")
    cols_dict["C"] = C

    df = pd.DataFrame(cols_dict)
    # Final cast of categorical columns that was impossible due to nan in numpy
    for i in range(Xcat.shape[1]):
        df[f"cat_col_{i}"] = df[f"cat_col_{i}"].astype("Int64")

    return df
