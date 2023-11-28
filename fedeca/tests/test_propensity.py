"""Tests for propensity score related functionality."""
import numpy as np
import pytest
from scipy.stats import binomtest

from fedeca.utils.survival_utils import CoxData

list_prop_treated = np.arange(0.1, 1, 0.1)


@pytest.mark.parametrize("prop_treated", list_prop_treated)
def test_constant_propensity(prop_treated: float):
    """Tests of data generation with constant propensity score."""
    data_gen = CoxData(
        n_samples=1,
        ndim=100,
        prop_treated=prop_treated,
        propensity="constant",
        seed=42,
    )

    n_samples = 1001
    _, _, _, treated, ps_scores = data_gen.generate_data(n_samples=n_samples)

    np.testing.assert_allclose(ps_scores, data_gen.prop_treated)

    n_treated_expected = int(n_samples * prop_treated)
    n_treated = treated.sum()
    # Constant propensity will use `random_treatment_allocation` which ensures
    # that `prop_treated` is respected, therefore check equality up to rounding.
    assert n_treated in (n_treated_expected, n_treated_expected + 1)


def test_linear_propensity():
    """Tests of data generation with linear propensity score."""
    prop_treated = 0.5
    data_gen = CoxData(
        n_samples=1,
        ndim=100,
        prop_treated=prop_treated,
        propensity="linear",
        seed=42,
    )

    n_samples = 1001
    _, _, _, treated, ps_scores = data_gen.generate_data(n_samples=n_samples)

    np.testing.assert_allclose(ps_scores.mean(), prop_treated, atol=0.05)

    conf_int = binomtest(treated.sum(), treated.size, prop_treated).proportion_ci()
    assert conf_int.low <= prop_treated <= conf_int.high
