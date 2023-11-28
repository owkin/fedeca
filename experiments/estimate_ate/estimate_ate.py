"""Simulation data for FedECA."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from fedeca import MatchingAjudsted, NaiveComparison, PooledIPTW
from fedeca.utils.survival_utils import CoxData

coxdata = CoxData(
    n_samples=250,
    cate="linear",
    seed=1234,
    percent_ties=None,
    ndim=20,
    propensity="linear",
    cov_corr=0.0,
    features_type="indep_gauss",
)
X, times, censoring, treat_alloc = coxdata.generate_data()
col_X = ["X_%i" % i for i in range(X.shape[1])]

data = np.concatenate(
    [X, times[:, np.newaxis], censoring[:, np.newaxis], treat_alloc[:, np.newaxis]],
    axis=1,
)

data = pd.DataFrame(
    data=data, columns=col_X + ["time", "event", "treatment_allocation"]
)
# define treatment allocation
treatment_allocation = "treatment_allocation"

print("Computing propensity weights on pooled data")
# Instantiate IPTW class
# We can specify the type of effect we want to estimate

iptw = PooledIPTW(
    treated_col=treatment_allocation,
    event_col="event",
    duration_col="time",
    effect="ATE",
)

# We can now estimate the treatment effect
iptw.fit(data)


naive_comparison = NaiveComparison(
    treated_col=treatment_allocation, event_col="event", duration_col="time"
)
naive_comparison.fit(data)

maic = MatchingAjudsted(
    treated_col=treatment_allocation,
    event_col="event",
    duration_col="time",
    effect="ATE",
)

data = data.drop("weights", axis=1)

mean_control = data.loc[data["treatment_allocation"] == 0, col_X].mean().to_frame().T
mean_control = mean_control.add_suffix(".mean")
sd_control = data.loc[data["treatment_allocation"] == 0, col_X].std().to_frame().T
sd_control = sd_control.add_suffix(".sd")

aggregated_control = pd.concat([mean_control, sd_control], axis=1)

maic.fit(
    data.loc[data["treatment_allocation"] == 1, :],
    aggregated_control,
    data.loc[data["treatment_allocation"] == 0, ["time", "event"]],
)

plt.clf()
ax = plt.subplot(111)

kmf_control = KaplanMeierFitter()
ax = kmf_control.fit(
    data.loc[data["treatment_allocation"] == 0, "time"],
    data.loc[data["treatment_allocation"] == 0, "event"],
    label="control",
    weights=iptw.weights_[data["treatment_allocation"] == 0],
).plot_survival_function(ax=ax)

kmf_treated = KaplanMeierFitter()
ax = kmf_treated.fit(
    data.loc[data["treatment_allocation"] == 1, "time"],
    data.loc[data["treatment_allocation"] == 1, "event"],
    label="treated",
    weights=iptw.weights_[data["treatment_allocation"] == 1],
).plot_survival_function(ax=ax)

plt.tight_layout()
plt.savefig("km.png")
