"""Test file for the BootstrapMixin."""
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from fedeca.utils.survival_utils import BootstrapMixin

seeds = np.arange(5)


@pytest.fixture
def test_settings():
    """Set up for the tests of the competitors class."""
    rng = np.random.default_rng(123)
    n_samples = 5
    n_features = 3

    X = rng.normal(size=(n_samples, n_features))
    data = pd.DataFrame(data=X, columns=[f"X_{i}" for i in range(n_features)])
    data["time"] = np.array([5, 6, 7, 8, 9])
    data["event"] = np.array([1, 1, 0, 1, 1])
    data["treatment_allocation"] = np.array([1, 0, 0, 1, 1])
    data["propensity_scores"] = np.array([0.55, 0.34, 0.76, 0.29, 0.32])
    data["weights"] = np.array([1.2, 1.4, 1.6, 1.8, 1.1])

    return data


class FakeModel(BootstrapMixin):
    """Fake model that supports bootstrapping."""

    def __init__(self, seed):
        """Initialize with seed."""
        self.rng = np.random.default_rng(seed)

    def point_estimate(self, data: pd.DataFrame) -> npt.ArrayLike:
        """Return a point estimate of the treatment effect."""
        return self.rng.random(size=2)


@pytest.mark.parametrize("seed", seeds)
def test_bootstrap(test_settings, seed: int):
    """Test BootstrapMixin."""
    model = FakeModel(seed)

    # Test resampling
    data = test_settings
    data_resampled = data.sample(
        data.shape[0], replace=True, random_state=np.random.default_rng(seed)
    )
    data_bootstrapped = model.bootstrap_sample(data, seed=seed)
    pd.testing.assert_frame_equal(data_resampled, data_bootstrapped)

    # Test bootstrapping
    std_est = model.bootstrap_std(test_settings, n_bootstrap=10)
    rng = np.random.default_rng(seed)
    std_true = np.std(rng.random(size=(10, 2)), axis=0)

    assert std_est is not None
    np.testing.assert_array_equal(std_est, std_true)
