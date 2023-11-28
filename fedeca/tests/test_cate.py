"""Tests for cate related functionality."""
import numpy as np
import pytest
from scipy.stats import binomtest

from fedeca.utils.survival_utils import CoxData

list_constant_cate = [0.5, 1.0, 1.5]
list_random_cate = ["linear", "random"]
list_features_type = ["cov_toeplitz", "cov_uniform", "indep_gauss"]


@pytest.mark.parametrize("cate", list_constant_cate)
def test_constant_cate(cate: float):
    """Tests of data generation with constant cate."""
    data_gen = CoxData(
        n_samples=1,
        ndim=10,
        cate=cate,
        seed=42,
    )

    n_samples = 1001
    data_gen.generate_data(n_samples=n_samples)

    cate_vector = data_gen.probability_treated
    assert cate_vector is not None
    np.testing.assert_allclose(cate_vector, cate)


@pytest.mark.parametrize("features_type", list_features_type)
@pytest.mark.parametrize("cate", list_random_cate)
def test_linear_cate(features_type, cate):
    """Tests of data generation with linear cate."""
    data_gen = CoxData(
        n_samples=1,
        ndim=10,
        cate=cate,
        features_type=features_type,
        seed=42,
    )

    n_samples = 1001
    data_gen.generate_data(n_samples=n_samples)

    cate_vector = data_gen.probability_treated
    assert cate_vector is not None
    # * linear_cate
    #   features are multivariate normal variables, whose linear combination is
    #   also normal, cate_vector is therefore log-normal, with median exp(0) = 1
    # * random_cate
    #   cate_vector is by definition log-normal, with median exp(0) = 1
    # test median
    conf_int = binomtest(np.sum(cate_vector > 1), cate_vector.size, 0.5).proportion_ci()
    assert conf_int.low <= 0.5 <= conf_int.high
