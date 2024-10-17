"""Script for YODA experiments."""

import logging
import warnings

import numpy as np
import pandas as pd
from lifelines.exceptions import StatisticalWarning

from fedeca import FedECA

logging.getLogger("substrafl").setLevel(logging.WARNING)
warnings.filterwarnings(action="ignore", category=StatisticalWarning)


def get_clients_sizes(df):
    """Get number of treatment types as number of clients."""
    return df["Treatment"].value_counts().sort_index(ascending=False).tolist()


def get_fedeca_fitter(df, variance_method="bootstrap"):
    """Get the FedECA fitter."""
    return FedECA(
        ndim=df.shape[1] - 3,
        treated_col="Treatment",
        event_col="Event",
        duration_col="Time",
        variance_method=variance_method,
        bootstrap_function="global",
        clients_sizes=get_clients_sizes(df),
        clients_names=["center0", "center1"],
        client_identifier="center",
    )


df_raw = pd.read_csv("path_to_imputed_data")

np.random.seed(42)

df3 = (
    df_raw.query(
        "(Trial == 'NCT02257736' & Treatment == 'Apalutamide_AA_P')"
        "| (Trial == 'NCT00887198' & Treatment == 'AA_P')"
    ).query("Endpoint == 'rPFS'")
).copy()
df3["Treatment"] = df3["Treatment"].eq("Apalutamide_AA_P").astype(int)
df3 = df3.drop(columns=["Patient ID", "Endpoint", "Trial"])

fed_iptw3 = get_fedeca_fitter(df3, variance_method="bootstrap")
fed_iptw3.fit(
    df3,
    n_clients=2,
    split_method="split_control_over_centers",
    split_method_kwargs={"treatment_info": "Treatment"},
    data_path=".\\data",
    backend_type="simu",
    bootstrap_seeds=np.random.randint(0, 2**31 - 1, 200).tolist(),
)
print(fed_iptw3.results_)

df5 = df_raw.query(
    "(Trial == 'NCT02257736' & Treatment == 'AA_P')"
    "| (Trial == 'NCT00887198' & Treatment == 'AA_P')"
).query("Endpoint == 'rPFS'")
df5["Treatment"] = df5["Trial"].eq("NCT02257736").astype(int)
df5 = df5.drop(columns=["Patient ID", "Endpoint", "Trial"])

fed_iptw5 = get_fedeca_fitter(df5, variance_method="bootstrap")
fed_iptw5.fit(
    df5,
    n_clients=2,
    split_method="split_control_over_centers",
    split_method_kwargs={"treatment_info": "Treatment"},
    data_path=".\\data",
    backend_type="simu",
    bootstrap_seeds=np.random.randint(0, 2**31 - 1, 200).tolist(),
)
print(fed_iptw5.results_)

df6 = df_raw.query(
    "(Trial == 'NCT02257736' & Treatment == 'AA_P')"
    "| (Trial == 'NCT00887198' & Treatment == 'Placebo_P')"
).query("Endpoint == 'rPFS'")
df6["Treatment"] = df6["Treatment"].eq("AA_P").astype(int)
df6 = df6.drop(columns=["Patient ID", "Endpoint", "Trial"])

fed_iptw6 = get_fedeca_fitter(df6, variance_method="bootstrap")
fed_iptw6.fit(
    df6,
    n_clients=2,
    split_method="split_control_over_centers",
    split_method_kwargs={"treatment_info": "Treatment"},
    data_path=".\\data",
    backend_type="simu",
    bootstrap_seeds=np.random.randint(0, 2**31 - 1, 200).tolist(),
)
print(fed_iptw6.results_)

df7 = df_raw.query(
    "(Trial == 'NCT02257736' & Treatment == 'Apalutamide_AA_P')"
    "| (Trial == 'NCT00887198' & Treatment == 'Placebo_P')"
).query("Endpoint == 'rPFS'")
df7["Treatment"] = df7["Treatment"].eq("Apalutamide_AA_P").astype(int)
df7 = df7.drop(columns=["Patient ID", "Endpoint", "Trial"])

fed_iptw7 = get_fedeca_fitter(df7, variance_method="bootstrap")
fed_iptw7.fit(
    df7,
    n_clients=2,
    split_method="split_control_over_centers",
    split_method_kwargs={"treatment_info": "Treatment"},
    data_path=".\\data",
    backend_type="simu",
    bootstrap_seeds=np.random.randint(0, 2**31 - 1, 200).tolist(),
)

print(fed_iptw7.results_)
