import pandas as pd
from fedeca.utils.survival_utils import CoxData
from fedeca import FedECA
# Let's generate 1000 data samples with 10 covariates
data = CoxData(seed=42, n_samples=1000, ndim=10)
df = data.generate_dataframe()

df = df.drop(columns=["propensity_scores"], axis=1)


fed_iptw = FedECA(ndim=10, treated_col="treatment", event_col="event", duration_col="time")
fed_iptw.fit(df, n_clients=4, split_method="split_control_over_centers", split_method_kwargs={"treatment_info": "treatment"}, data_path="./data", variance_method="robust", backend_type="simu")
print(fed_iptw.results_)