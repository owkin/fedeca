"""Test file for the competitors."""
import numpy as np
import pandas
import pytest
from indcomp import MAIC
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression

from fedeca import MatchingAjudsted, NaiveComparison, PooledIPTW


@pytest.fixture
def test_settings():
    """Set up for the tests of the competitors class."""
    rng = np.random.default_rng(123)
    n_samples = 5
    n_features = 3

    X = rng.normal(size=(n_samples, n_features))
    data = pandas.DataFrame(data=X, columns=[f"X_{i}" for i in range(n_features)])
    data["time"] = np.array([5, 6, 7, 8, 9])
    data["event"] = np.array([1, 1, 0, 1, 1])
    data["treatment_allocation"] = np.array([1, 0, 0, 1, 1])
    data["propensity_scores"] = np.array([0.55, 0.34, 0.76, 0.29, 0.32])
    data["weights"] = np.array([1.2, 1.4, 1.6, 1.8, 1.1])

    return data


def test_naive_comparison(test_settings):
    """Tests for naive comparison class."""
    data = test_settings

    naive_comparison = NaiveComparison(
        treated_col="treatment_allocation", event_col="event", duration_col="time"
    )
    naive_comparison.fit(data)

    cox_model = CoxPHFitter()
    cox_model.fit(
        data[["time", "event", "treatment_allocation"]],
        "time",
        "event",
    )

    pandas.testing.assert_frame_equal(
        left=cox_model.summary, right=naive_comparison.results_
    )


def test_iptw(test_settings):
    """Tests for the PooledIPTW class."""
    data = test_settings

    # tests weights computation

    data = data.drop(["weights"], axis=1)
    logreg = LogisticRegression(solver="lbfgs", penalty=None)
    mask_col = data.columns.isin(
        ["treatment_allocation", "event", "time", "propensity_scores"]
    )
    logreg.fit(np.array(data.loc[:, ~mask_col]), data["treatment_allocation"])

    propensity_scores = logreg.predict_proba(np.array(data.loc[:, ~mask_col]))[:, 1]
    weights = np.divide(1, propensity_scores) * np.array(data["treatment_allocation"])
    weights += np.divide(1, 1 - propensity_scores) * (
        1 - np.array(data["treatment_allocation"])
    )

    pooled_iptw = PooledIPTW(
        treated_col="treatment_allocation",
        event_col="event",
        duration_col="time",
        variance_method="robust",
    )
    pooled_iptw.fit(data)

    np.testing.assert_allclose(
        pooled_iptw.weights_,
        weights,
    )

    # tests that weighted cox is well performed
    data = test_settings

    results = pooled_iptw._estimate_effect(
        data.drop(["weights"], axis=1), data["weights"]
    )

    weighted_cox = CoxPHFitter()
    weighted_cox.fit(
        data[["time", "event", "treatment_allocation", "weights"]],
        "time",
        "event",
        weights_col="weights",
        robust=True,
    )

    pandas.testing.assert_frame_equal(left=weighted_cox.summary, right=results.summary)


def test_maic(test_settings):
    """Test for MAIC class."""
    data = test_settings
    data = data.drop(["weights"], axis=1)
    treated_col = "treatment_allocation"
    maic = MatchingAjudsted(
        treated_col=treated_col,
        event_col="event",
        duration_col="time",
        variance_method="robust",
    )
    maic.fit(data)

    # MAIC model tested here
    # https://github.com/AidanCooper/indcomp/blob/main/tests/test_maic.py
    df_agg = data.groupby(treated_col)
    df_agg = df_agg[["X_0", "X_1", "X_2"]].agg(["mean", "std"])
    df_agg.columns = [".".join(x) for x in df_agg.columns]
    targets = df_agg.loc[[0]]

    matching_dict = {}
    for col in ["X_0", "X_1", "X_2"]:
        matching_dict[col + ".mean"] = ("mean", col)
        matching_dict[col + ".std"] = ("std", col, col + ".mean")

    true_maic = MAIC(
        df_index=data.loc[data[treated_col] == 1],
        df_target=targets,
        match=matching_dict,
    )
    true_maic.calc_weights()

    true_weights = true_maic.weights_

    data.loc[data[treated_col].eq(1), "weights"] = true_weights
    data.loc[~data[treated_col].eq(1), "weights"] = 1

    weighted_cox = CoxPHFitter()
    weighted_cox.fit(
        data[["time", "event", treated_col, "weights"]],
        "time",
        "event",
        weights_col="weights",
        robust=True,
    )

    pandas.testing.assert_frame_equal(left=weighted_cox.summary, right=maic.results_)
