import pandas as pd
from fedeca.utils.survival_utils import CoxData
from fedeca import FedECA
import os
# Let's generate 1000 data samples with 10 covariates
data = CoxData(seed=42, n_samples=1000, ndim=10)
df = data.generate_dataframe()

df = df.drop(columns=["propensity_scores"], axis=1)

os.system("rm /Users/jterrail/Desktop/workflow.txt")
with open("/Users/jterrail/Desktop/workflow.txt", "w") as f:
    f.write("<bloc>\n<name>FedECA</name>\n")
fed_iptw = FedECA(ndim=10, treated_col="treatment", event_col="event", duration_col="time", num_rounds_list=[2, 3], variance_method="naïve")
fed_iptw.fit(df, n_clients=4, split_method="split_control_over_centers", split_method_kwargs={"treatment_info": "treatment"}, data_path="./data", backend_type="simu")
print(fed_iptw.results_)
with open("/Users/jterrail/Desktop/workflow.txt", "a") as f:
    f.write("</bloc>\n")
