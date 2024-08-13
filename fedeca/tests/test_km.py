"""Tests for the survival utils related to Kaplan Meier."""
import numpy as np
from lifelines import KaplanMeierFitter

from fedeca.utils.survival_utils import compute_events_statistics, km_curve


def test_compute_events_statistics():
    """Test the computation of events statistics."""
    times = np.array([0.0, 3.2, 3.2, 5.5, 5.5, 10.0])
    events = np.array([True, True, False, True, True, False])
    unique_times_gt = np.array([0.0, 3.2, 5.5, 10.0])
    num_death_at_times_gt = np.array([1, 1, 2, 0])
    num_at_risk_at_times_gt = np.array([6, 5, 3, 1])

    # Check that the result is correct
    t, n, d = compute_events_statistics(times, events)
    assert np.allclose(t, unique_times_gt)
    assert np.allclose(n, num_at_risk_at_times_gt)
    assert np.allclose(d, num_death_at_times_gt)
    # Check that the result is invariant by permutation
    p = np.random.permutation(times.size)
    t_p, n_p, d_p = compute_events_statistics(times[p], events[p])
    for (a, b) in zip([t_p, n_p, d_p], [t, n, d]):
        assert np.allclose(a, b)


def test_km_curve():
    """Test the computation of the Kaplan Meier curve."""
    rng = np.random.RandomState(42)
    num_samples = 100
    times = rng.randint(0, high=21, size=(num_samples,))
    events = rng.rand(num_samples) > 0.5
    # Get KM curve
    t, n, d = compute_events_statistics(times, events)
    grid, s, _ = km_curve(t, n, d, tmax=20)
    # Compute the one from lifelines
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events)
    s_gt = kmf.survival_function_["KM_estimate"].to_numpy()
    grid_gt = kmf.survival_function_.index.to_numpy()
    assert np.allclose(grid_gt, grid)
    assert np.allclose(s_gt, s)
